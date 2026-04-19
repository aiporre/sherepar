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
 * Metadata returned alongside deformed geometry.
 */
struct DeformMeta {
    std::string template_mesh_path;    ///< Path to the source OBJ mesh
    std::string method;                ///< Algorithm name as string
    std::vector<int> handle_ids;       ///< Vertex indices used as handles
    /// Target positions corresponding to each handle (flat: [x0,y0,z0, x1,y1,z1, ...])
    std::vector<double> target_positions;
    std::vector<int> roi_ids;          ///< Region-of-interest vertex indices (empty = full mesh)
    double alpha;                      ///< SRE_ARAP smoothness weight
    int max_iter;                      ///< Maximum deformation iterations
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
 * @returns Tuple (vertices, faces, meta) where:
 *   - vertices is a flat array of doubles [x0,y0,z0, x1,y1,z1, ...], size 3*N
 *   - faces    is a flat array of ints   [a0,b0,c0, a1,b1,c1, ...], size 3*M
 *   - meta     is a DeformMeta struct
 *
 * @throws std::runtime_error on any error (file not found, bad indices, CGAL failure).
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

} // namespace graphop

#endif // GRAPHOP_DEFORMATION_H