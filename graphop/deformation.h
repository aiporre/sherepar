#ifndef GRAPHOP_DEFORMATION_H
#define GRAPHOP_DEFORMATION_H

/**
 * deformation.h
 *
 * CGAL Surface_mesh_deformation backend for pmConv Stage 1.
 * Supports ORIGINAL_ARAP, SPOKES_AND_RIMS, and SRE_ARAP algorithms.
 *
 * Lives under graphop/ as part of the sherepar project.
 */

#include <string>
#include <vector>
#include <tuple>
#include <stdexcept>

namespace graphop {

/**
 * Deformation algorithm selection.
 * Maps onto CGAL::Deformation_algorithm_tag values.
 */
enum class DeformMethod {
    ORIGINAL_ARAP,   ///< Standard ARAP
    SPOKES_AND_RIMS, ///< Spokes-and-rims ARAP variant
    SRE_ARAP         ///< Smooth Rotation Enhanced ARAP
};

/**
 * Per-handle rotation specification for deform_surface_with_angles().
 *
 * vertex_id  : 0-based index of the center vertex.
 * angle      : Rotation angle in radians around the surface normal at vertex_id.
 * ring_size  : Euclidean radius. Every mesh vertex whose distance from vertex_id
 *              is <= ring_size becomes a control vertex, rotated by angle around
 *              the surface normal at vertex_id.
 */
struct HandleTransform {
    int    vertex_id;
    double angle;
    double ring_size;
    std::vector<double> center_coords; // optional 3D coordinates for the rotation center; if empty, use vertex position
};

/**
 * Metadata returned alongside deformed geometry.
 *
 * For deform_surface()             : transform_* vectors are empty.
 * For deform_surface_with_angles() : transform_* vectors record the original
 *                                    HandleTransform inputs.
 */
struct DeformMeta {
    std::string template_mesh_path;    ///< Path to the source OBJ mesh
    std::string method;                ///< Algorithm name as string
    std::vector<int>    handle_ids;    ///< Vertex indices used as handles
    std::vector<double> target_positions; ///< Flat [x,y,z,...] target per handle
    std::vector<int>    roi_ids;       ///< ROI vertex indices (empty = full mesh)
    double alpha;                      ///< SRE_ARAP smoothness weight
    int    max_iter;                   ///< Maximum deformation iterations

    // Fields set only by deform_surface_with_angles
    std::vector<int>    transform_center_ids;
    std::vector<double> transform_angles;
    std::vector<double> transform_ring_sizes;
    std::vector<double> transform_center_coords; ///< Flat [x,y,z,...] center used per transform
};

/**
 * Deform a triangulated surface mesh loaded from an OBJ file.
 *
 * @param mesh_path         Path to the input OBJ mesh file.
 * @param handle_ids        0-based vertex indices to use as positional handles.
 * @param target_positions  Target 3-D positions for each handle, same order as
 *                          handle_ids; length must be 3*handle_ids.size().
 * @param roi_ids           Optional ROI vertex indices.  Empty means the whole mesh.
 * @param method            Deformation algorithm to use (default: SRE_ARAP).
 * @param alpha             SRE_ARAP smoothness parameter (ignored for other methods).
 * @param max_iter          Maximum ARAP iterations.
 *
 * @returns Tuple (vertices, faces, meta).
 * @throws std::runtime_error on any error.
 */
std::tuple<std::vector<double>, std::vector<int>, DeformMeta>
deform_surface(
    const std::string& mesh_path,
    const std::vector<int>& handle_ids,
    const std::vector<double>& target_positions,
    const std::vector<int>& roi_ids = {},
    DeformMethod method = DeformMethod::SRE_ARAP,
    double alpha = 0.02,
    int max_iter = 50
);

/**
 * Deform a surface using per-handle rotation specifications.
 *
 * For each HandleTransform t:
 *   1. Center c = position of vertex t.vertex_id.
 *   2. Rotation axis k = area-weighted surface normal at t.vertex_id.
 *   3. Every vertex v with ||v - c|| <= t.ring_size becomes a control vertex;
 *      its target is c + Eigen::Quaterniond(AngleAxisd(t.angle, k)) * (v - c).
 *
 * @param mesh_path          Path to the input OBJ mesh file.
 * @param handle_transforms  Per-handle rotation specifications.
 * @param roi_ids            Optional ROI.  Empty = whole mesh.
 * @param method             Deformation algorithm (default: SRE_ARAP).
 * @param alpha              SRE_ARAP smoothness weight.
 * @param max_iter           Maximum ARAP iterations.
 *
 * @returns Tuple (vertices, faces, meta).
 * @throws std::runtime_error on any error.
 */
std::tuple<std::vector<double>, std::vector<int>, DeformMeta>
deform_surface_with_angles(
    const std::string& mesh_path,
    const std::vector<HandleTransform>& handle_transforms,
    const std::vector<int>& roi_ids = {},
    DeformMethod method = DeformMethod::SRE_ARAP,
    double alpha = 0.02,
    int max_iter = 50
);

} // namespace graphop

#endif // GRAPHOP_DEFORMATION_H