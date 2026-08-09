use std::{cmp::Ordering, collections::BinaryHeap, error::Error, fmt, path::PathBuf, process};

use clap::Parser;
use glam::{DVec4, Mat3, Quat, Vec3, vec3};
use vortex_project::{PartTransform, Project};

#[derive(Debug, Clone, Copy)]
pub struct BoxTransform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

#[derive(Debug)]
pub enum MeshToBoxesError {
    InvalidIndexCount,
    NonPositiveTolerance,
    IndexOutOfBounds { index: u32, vertex_count: usize },
}

impl fmt::Display for MeshToBoxesError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIndexCount => {
                write!(formatter, "mesh indices must be a multiple of three")
            }
            Self::NonPositiveTolerance => {
                write!(formatter, "surface-error tolerance must be positive")
            }
            Self::IndexOutOfBounds {
                index,
                vertex_count,
            } => write!(
                formatter,
                "mesh index {index} is outside the {vertex_count} available vertices"
            ),
        }
    }
}

impl Error for MeshToBoxesError {}

struct Triangle {
    vertices: [Vec3; 3],
    // Fraction of one original input triangle represented by this triangle
    weight: f64,
}

struct Cluster {
    triangles: Vec<Triangle>,
    obb: BoxTransform,
    surface_error: f32,
}

struct WorkItem {
    cluster: Cluster,
    sequence: usize,
}

impl PartialEq for WorkItem {
    fn eq(&self, other: &Self) -> bool {
        self.sequence == other.sequence
    }
}

impl Eq for WorkItem {}

impl PartialOrd for WorkItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WorkItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cluster
            .surface_error
            .total_cmp(&other.cluster.surface_error)
            // Prefer earlier clusters when their triangle counts match, so
            // conversion remains deterministic.
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

pub fn mesh_to_boxes(
    vertices: &[Vec3],
    indices: &[u32],
    max_relative_diagonal: f32,
) -> Result<Vec<BoxTransform>, MeshToBoxesError> {
    if !indices.len().is_multiple_of(3) {
        return Err(MeshToBoxesError::InvalidIndexCount);
    }
    if !max_relative_diagonal.is_finite() || max_relative_diagonal <= 0.0 {
        return Err(MeshToBoxesError::NonPositiveTolerance);
    }

    let mut triangles = Vec::with_capacity(indices.len() / 3);
    for triangle_indices in indices.chunks_exact(3) {
        let [first, second, third] = [
            triangle_indices[0],
            triangle_indices[1],
            triangle_indices[2],
        ]
        .map(|index| {
            vertices
                .get(index as usize)
                .copied()
                .ok_or(MeshToBoxesError::IndexOutOfBounds {
                    index,
                    vertex_count: vertices.len(),
                })
        });
        triangles.push(Triangle {
            vertices: [first?, second?, third?],
            weight: 1.0,
        });
    }

    if triangles.is_empty() {
        return Ok(Vec::new());
    }

    let triangle_count = triangles.len();
    let mut clusters = BinaryHeap::new();
    let root = make_cluster(triangles);
    let root_diagonal = root.obb.scale.length();
    let max_surface_error = root_diagonal * max_relative_diagonal;
    eprintln!(
        "Splitting until box surface error <= {:.2}% of the mesh diagonal...",
        max_relative_diagonal * 100.0
    );
    clusters.push(WorkItem {
        cluster: root,
        sequence: 0,
    });
    let mut next_sequence = 1;
    let mut next_progress_percent = 1;
    let triangle_weight = triangle_count as f64;
    let mut finalized_weight = 0.0;
    let mut finished = Vec::new();

    while let Some(WorkItem { cluster, .. }) = clusters.pop() {
        // A box is final only when its surface closely follows the mesh. This
        // keeps an exact cube as one part while refining curved, concave, and
        // partially filled bounding boxes.
        if cluster.surface_error <= max_surface_error {
            finalized_weight += cluster
                .triangles
                .iter()
                .map(|triangle| triangle.weight)
                .sum::<f64>();
            finished.push(cluster);

            let percent_complete = (finalized_weight * 100.0 / triangle_weight).floor() as usize;
            while percent_complete >= next_progress_percent {
                eprint!("    {}%\r", next_progress_percent);
                next_progress_percent += 1;
            }
            continue;
        }

        // Always refine the largest box that is still over budget.
        let (a, b) = split_cluster(cluster);

        clusters.push(WorkItem {
            cluster: a,
            sequence: next_sequence,
        });
        next_sequence += 1;
        clusters.push(WorkItem {
            cluster: b,
            sequence: next_sequence,
        });
        next_sequence += 1;
    }

    eprintln!("======================");
    eprintln!("Finished with {} boxes.", finished.len());
    Ok(finished.into_iter().map(|cluster| cluster.obb).collect())
}

fn make_cluster(triangles: Vec<Triangle>) -> Cluster {
    let mut obb = fit_obb(&triangles);
    let mut surface_error = box_surface_error(&triangles, obb);

    // PCA is ambiguous for symmetric meshes such as cubes. Triangle face
    // normals provide an alternate orientation that preserves their edges.
    if triangles.len() >= 4
        && let Some(basis) = face_aligned_basis(&triangles)
    {
        let candidate = fit_obb_in_basis(&triangles, basis);
        let candidate_error = box_surface_error(&triangles, candidate);
        if candidate_error < surface_error {
            obb = candidate;
            surface_error = candidate_error;
        }
    }

    Cluster {
        triangles,
        obb,
        surface_error,
    }
}

fn split_cluster(cluster: Cluster) -> (Cluster, Cluster) {
    if cluster.triangles.len() == 1 {
        let [left, right] = split_triangle(cluster.triangles.into_iter().next().unwrap());
        return (make_cluster(vec![left]), make_cluster(vec![right]));
    }

    let axes = Mat3::from_quat(cluster.obb.rotation);

    // Split along the longest dimension of the current box.
    let scale = cluster.obb.scale;

    let axis = if scale.x >= scale.y && scale.x >= scale.z {
        axes.x_axis
    } else if scale.y >= scale.z {
        axes.y_axis
    } else {
        axes.z_axis
    };

    let mut triangles = cluster.triangles;
    let middle = triangles.len() / 2;

    // Partition by the median centroid without sorting the whole cluster.
    // This is linear-time on average, whereas a full sort is O(n log n).
    triangles.select_nth_unstable_by(middle, |a, b| {
        let a_center = (a.vertices[0] + a.vertices[1] + a.vertices[2]) / 3.0;

        let b_center = (b.vertices[0] + b.vertices[1] + b.vertices[2]) / 3.0;

        let a_proj = a_center.dot(axis);
        let b_proj = b_center.dot(axis);

        a_proj.total_cmp(&b_proj)
    });

    let right = triangles.split_off(middle);
    let left = triangles;

    (make_cluster(left), make_cluster(right))
}

fn split_triangle(triangle: Triangle) -> [Triangle; 2] {
    let [a, b, c] = triangle.vertices;
    let ab = a.distance_squared(b);
    let bc = b.distance_squared(c);
    let ca = c.distance_squared(a);

    if ab >= bc && ab >= ca {
        let midpoint = (a + b) * 0.5;
        [
            Triangle {
                vertices: [a, midpoint, c],
                weight: triangle.weight * 0.5,
            },
            Triangle {
                vertices: [midpoint, b, c],
                weight: triangle.weight * 0.5,
            },
        ]
    } else if bc >= ca {
        let midpoint = (b + c) * 0.5;
        [
            Triangle {
                vertices: [b, midpoint, a],
                weight: triangle.weight * 0.5,
            },
            Triangle {
                vertices: [midpoint, c, a],
                weight: triangle.weight * 0.5,
            },
        ]
    } else {
        let midpoint = (c + a) * 0.5;
        [
            Triangle {
                vertices: [c, midpoint, b],
                weight: triangle.weight * 0.5,
            },
            Triangle {
                vertices: [midpoint, a, b],
                weight: triangle.weight * 0.5,
            },
        ]
    }
}

fn fit_obb(triangles: &[Triangle]) -> BoxTransform {
    if triangles.is_empty() {
        return BoxTransform {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ZERO,
        };
    }

    let point_count = triangles.len() * 3;
    let mean = triangles.iter().flat_map(|t| t.vertices).sum::<Vec3>() / point_count as f32;

    let covariance = covariance_matrix(triangles, mean);

    // Principal axis.
    let x = principal_eigenvector(covariance, Vec3::X);

    // Remove contribution of first eigenvector and find second.
    let lambda_x = x.dot(covariance * x);

    let deflated = covariance - outer_product(x, x) * lambda_x;

    let mut y = principal_eigenvector(deflated, Vec3::Y);

    // Gram-Schmidt, mostly to counter floating point drift.
    y -= x * y.dot(x);

    if y.length_squared() < 1e-10 {
        y = perpendicular(x);
    } else {
        y = y.normalize();
    }

    let mut z = x.cross(y).normalize();

    // Recalculate Y so the basis is guaranteed orthogonal.
    y = z.cross(x).normalize();

    // Ensure right-handed coordinate system.
    if x.cross(y).dot(z) < 0.0 {
        z = -z;
    }

    let basis = Mat3::from_cols(x, y, z);

    fit_obb_in_basis(triangles, basis)
}

fn fit_obb_in_basis(triangles: &[Triangle], basis: Mat3) -> BoxTransform {
    let point_count = triangles.len() * 3;
    let mean = triangles.iter().flat_map(|t| t.vertices).sum::<Vec3>() / point_count as f32;

    let inverse_basis = basis.transpose();

    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);

    for triangle in triangles {
        for &point in &triangle.vertices {
            let local = inverse_basis * (point - mean);

            min = min.min(local);
            max = max.max(local);
        }
    }

    let local_center = (min + max) * 0.5;
    let scale = max - min;

    let world_center = mean + basis * local_center;

    BoxTransform {
        position: world_center,
        rotation: Quat::from_mat3(&basis),
        scale,
    }
}

fn face_aligned_basis(triangles: &[Triangle]) -> Option<Mat3> {
    let first_normal = triangles
        .iter()
        .map(triangle_normal)
        .find(|normal| normal.length_squared() > 1e-12)?;
    let x = first_normal.normalize();

    for triangle in triangles {
        let normal = triangle_normal(triangle);
        let mut y = normal - x * normal.dot(x);
        if y.length_squared() <= 1e-12 {
            continue;
        }

        y = y.normalize();
        let z = x.cross(y);
        return Some(Mat3::from_cols(x, y, z));
    }

    None
}

fn triangle_normal(triangle: &Triangle) -> Vec3 {
    let [a, b, c] = triangle.vertices;
    (b - a).cross(c - a)
}

fn box_surface_error(triangles: &[Triangle], obb: BoxTransform) -> f32 {
    let basis = Mat3::from_quat(obb.rotation);
    let half_scale = obb.scale * 0.5;
    let mut worst_distance_squared = 0.0_f32;

    for x in [-1.0, 0.0, 1.0] {
        for y in [-1.0, 0.0, 1.0] {
            for z in [-1.0, 0.0, 1.0] {
                if x == 0.0 && y == 0.0 && z == 0.0 {
                    continue;
                }

                let point = obb.position + basis * (half_scale * Vec3::new(x, y, z));
                let distance_squared = triangles
                    .iter()
                    .map(|triangle| point_triangle_distance_squared(point, triangle.vertices))
                    .fold(f32::INFINITY, f32::min);

                worst_distance_squared = worst_distance_squared.max(distance_squared);
            }
        }
    }

    worst_distance_squared.sqrt()
}

fn point_triangle_distance_squared(point: Vec3, [a, b, c]: [Vec3; 3]) -> f32 {
    let ab = b - a;
    let ac = c - a;
    let ap = point - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return ap.length_squared();
    }

    let bp = point - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return bp.length_squared();
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (ap - ab * v).length_squared();
    }

    let cp = point - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return cp.length_squared();
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (ap - ac * w).length_squared();
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let bc = c - b;
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (bp - bc * w).length_squared();
    }

    let denominator = 1.0 / (va + vb + vc);
    let v = vb * denominator;
    let w = vc * denominator;
    (a + ab * v + ac * w - point).length_squared()
}

fn covariance_matrix(triangles: &[Triangle], mean: Vec3) -> Mat3 {
    let mut xx = 0.0;
    let mut xy = 0.0;
    let mut xz = 0.0;
    let mut yy = 0.0;
    let mut yz = 0.0;
    let mut zz = 0.0;

    for triangle in triangles {
        for &point in &triangle.vertices {
            let d = point - mean;

            xx += d.x * d.x;
            xy += d.x * d.y;
            xz += d.x * d.z;

            yy += d.y * d.y;
            yz += d.y * d.z;

            zz += d.z * d.z;
        }
    }

    let n = (triangles.len() * 3) as f32;

    Mat3::from_cols(
        Vec3::new(xx, xy, xz) / n,
        Vec3::new(xy, yy, yz) / n,
        Vec3::new(xz, yz, zz) / n,
    )
}

fn principal_eigenvector(matrix: Mat3, initial: Vec3) -> Vec3 {
    let mut v = initial.normalize();

    for _ in 0..32 {
        let next = matrix * v;

        if next.length_squared() < 1e-12 {
            break;
        }

        v = next.normalize();
    }

    v
}

fn outer_product(a: Vec3, b: Vec3) -> Mat3 {
    Mat3::from_cols(a * b.x, a * b.y, a * b.z)
}

fn perpendicular(v: Vec3) -> Vec3 {
    // pick the cardinal axis least parallel to v
    let candidate = if v.x.abs() <= v.y.abs() && v.x.abs() <= v.z.abs() {
        Vec3::X
    } else if v.y.abs() <= v.z.abs() {
        Vec3::Y
    } else {
        Vec3::Z
    };

    v.cross(candidate).normalize()
}

#[derive(Debug, Parser)]
#[command(about = "Convert an OBJ mesh into fitted box parts")]
struct Cli {
    /// Source OBJ mesh to convert.
    input: PathBuf,

    /// Optional output JSON project. Defaults to --project, or INPUT with a
    /// .json extension when no project is supplied.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Existing JSON project to extend with the generated parts.
    #[arg(long)]
    project: Option<PathBuf>,

    /// Mesh translation, as X Y Z.
    #[arg(long, num_args = 3, value_names = ["X", "Y", "Z"], value_parser = parse_finite_f32, allow_hyphen_values = true)]
    position: Option<Vec<f32>>,

    /// Mesh scale, as positive X Y Z values.
    #[arg(long, num_args = 1..=3, value_names = ["X", "Y", "Z"], value_parser = parse_positive_f32, allow_hyphen_values = true)]
    scale: Option<Vec<f32>>,

    /// Maximum box-to-mesh surface error as a fraction of the mesh diagonal. (Lower = higher quality = more parts.)
    #[arg(short, long, visible_alias = "merror", default_value_t = 0.01, value_parser = parse_positive_f32)]
    max_relative_surface_error: f32,

    #[arg(long, default_value = "FF0000", value_parser = parse_color)]
    color: DVec4,

    #[arg(long, default_value = "Plastic", value_parser = parse_material)]
    material: String,
}

fn parse_finite_f32(value: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| "must be a number".to_owned())?;

    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err("must be finite".to_owned())
    }
}

fn parse_positive_f32(value: &str) -> Result<f32, String> {
    let parsed = parse_finite_f32(value)?;

    if parsed.is_finite() && parsed > 0.0 {
        Ok(parsed)
    } else {
        Err("must be a positive finite number".to_owned())
    }
}

fn parse_color(value: &str) -> Result<DVec4, String> {
    let value = value.strip_prefix('#').unwrap_or(value);
    if value.len() != 6 && value.len() != 8 {
        return Err("must be RRGGBB or RRGGBBAA hexadecimal".to_owned());
    }

    let component = |start| {
        u8::from_str_radix(&value[start..start + 2], 16)
            .map(|component| f64::from(component) / 255.0)
            .map_err(|_| "must be RRGGBB or RRGGBBAA hexadecimal".to_owned())
    };

    Ok(DVec4::new(
        component(0)?,
        component(2)?,
        component(4)?,
        if value.len() == 8 { component(6)? } else { 1.0 },
    ))
}

fn parse_material(value: &str) -> Result<String, String> {
    if value.trim().is_empty() {
        Err("must not be empty".to_owned())
    } else {
        Ok(value.to_owned())
    }
}

fn vec3_argument(values: Option<&[f32]>, default: Vec3) -> Vec3 {
    match values {
        Some([x, y, z]) => vec3(*x, *y, *z),
        None => default,
        Some([x]) => Vec3::splat(*x),
        Some(_) => unreachable!("clap errors out before this"),
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let path = cli.input;
    let position = vec3_argument(cli.position.as_deref(), Vec3::ZERO);
    let scale = vec3_argument(cli.scale.as_deref(), Vec3::ONE);

    let (models, _) = tobj::load_obj(&path, &tobj::GPU_LOAD_OPTIONS)?;
    eprintln!("Loaded {} mesh(es).", models.len());

    let mut parts = vec![];

    for (model_index, model) in models.into_iter().enumerate() {
        let mut position_chunks = model.mesh.positions.chunks_exact(3);
        let vec3s: Vec<Vec3> = position_chunks
            .by_ref()
            .map(|chunk| vec3(chunk[0], chunk[1], chunk[2]) * scale)
            .collect();
        if !position_chunks.remainder().is_empty() {
            return Err("mesh position data must be a multiple of three".into());
        }
        let indices = model.mesh.indices;

        eprintln!(
            "Converting mesh {} ({} triangles)...",
            model_index + 1,
            indices.len() / 3
        );
        let boxes = mesh_to_boxes(vec3s.as_slice(), &indices, cli.max_relative_surface_error)?;
        eprintln!(
            "Mesh {} contributed {} parts.",
            model_index + 1,
            boxes.len()
        );
        parts.extend(boxes);
    }

    let mut project = match &cli.project {
        Some(existing_project) => Project::load(existing_project)?,
        None => Project::new(),
    };
    project.add_parts(parts.into_iter().map(|part| PartTransform {
        position: part.position.as_dvec3() + position.as_dvec3(),
        rotation: part.rotation.as_dquat(),
        scale: part.scale.as_dvec3(),
        color: cli.color,
        material: cli.material.clone(),
    }));

    let output = match cli.output {
        Some(output) => output,
        None => cli.project.unwrap_or_else(|| path.with_extension("json")),
    };
    project.write(output)?;
    Ok(())
}

// entering vibe territory
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_mesh_input() {
        let vertices = [vec3(0.0, 0.0, 0.0)];

        assert!(matches!(
            mesh_to_boxes(&vertices, &[0, 1, 2], 0.01),
            Err(MeshToBoxesError::IndexOutOfBounds { .. })
        ));
        assert!(matches!(
            mesh_to_boxes(&vertices, &[0, 0], 0.01),
            Err(MeshToBoxesError::InvalidIndexCount)
        ));
    }

    #[test]
    fn keeps_a_planar_rectangle_as_one_part() {
        // Four triangles in the Z=0 plane. Every fitted box has zero volume,
        // which used to make the first split fail the volume-gain check.
        let vertices = [
            vec3(0.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(2.0, 0.0, 0.0),
            vec3(3.0, 0.0, 0.0),
            vec3(2.0, 1.0, 0.0),
            vec3(4.0, 0.0, 0.0),
            vec3(5.0, 0.0, 0.0),
            vec3(4.0, 1.0, 0.0),
            vec3(6.0, 0.0, 0.0),
            vec3(7.0, 0.0, 0.0),
            vec3(6.0, 1.0, 0.0),
        ];
        let indices: Vec<u32> = (0..12).collect();

        let boxes = mesh_to_boxes(&vertices, &indices, 0.5).unwrap();

        assert_eq!(boxes.len(), 1);
    }

    #[test]
    fn subdivides_a_poorly_fitting_triangle() {
        let vertices = [
            vec3(0.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(0.0, 1.0, 0.0),
        ];
        let indices: Vec<u32> = (0..3).collect();
        let boxes = mesh_to_boxes(&vertices, &indices, 0.01).unwrap();

        assert!(boxes.len() > 1);
    }

    #[test]
    fn keeps_an_exact_cube_as_one_part() {
        let vertices = [
            vec3(-1.0, -1.0, -1.0),
            vec3(1.0, -1.0, -1.0),
            vec3(1.0, 1.0, -1.0),
            vec3(-1.0, 1.0, -1.0),
            vec3(-1.0, -1.0, 1.0),
            vec3(1.0, -1.0, 1.0),
            vec3(1.0, 1.0, 1.0),
            vec3(-1.0, 1.0, 1.0),
        ];
        let indices = [
            0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7,
            3, 3, 7, 4, 3, 4, 0,
        ];

        let boxes = mesh_to_boxes(&vertices, &indices, 0.01).unwrap();

        assert_eq!(boxes.len(), 1);
    }
}
