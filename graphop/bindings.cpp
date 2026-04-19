/**
 * bindings.cpp
 *
 * pybind11 bindings for the graphop deformation backend.
 *
 * Exposes deform_surface() to Python as part of the `graphop` extension module.
 * The Python function signature mirrors the design in the problem statement:
 *
 *   V_new, F, meta = graphop.deform_surface(
 *       mesh_path,
 *       handle_ids,
 *       target_positions,
 *       roi_ids=[],
 *       method="sre_arap",
 *       alpha=0.02,
 *       max_iter=50,
 *   )
 *
 * Returns:
 *   V_new  – NumPy array [N, 3] float64
 *   F      – NumPy array [M, 3] int32
 *   meta   – Python dict with all deformation metadata
 */

#include "deformation.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace graphop;

// utils 

static DeformMethod parse_method(const std::string& name)
{
    if (name == "sre_arap")         return DeformMethod::SRE_ARAP;
    if (name == "original_arap")    return DeformMethod::ORIGINAL_ARAP;
    if (name == "spokes_and_rims")  return DeformMethod::SPOKES_AND_RIMS;
    throw std::invalid_argument(
        "Unknown method '" + name + "'. Choose from: "
        "'sre_arap', 'original_arap', 'spokes_and_rims'.");
}


// --------
//  Python-facing wrapper 
// --------


static py::tuple py_deform_surface(
    const std::string& mesh_path,
    const std::vector<int>& handle_ids,
    py::array_t<double> target_positions_arr,
    const std::vector<int>& roi_ids,
    const std::string& method_str,
    double alpha,
    int max_iter)
{
    // Flatten target_positions from (H,3) or (3H,) to std::vector<double>
    auto buf = target_positions_arr.request();
    if (buf.ndim == 2) {
        if (buf.shape[0] != (ssize_t)handle_ids.size() || buf.shape[1] != 3)
            throw std::invalid_argument(
                "target_positions shape must be (len(handle_ids), 3) or (3*len(handle_ids),)");
    } else if (buf.ndim == 1) {
        if (buf.shape[0] != (ssize_t)(handle_ids.size() * 3))
            throw std::invalid_argument(
                "target_positions flat length must be 3 * len(handle_ids)");
    } else {
        throw std::invalid_argument("target_positions must be 1-D or 2-D");
    }

    const double* data = static_cast<const double*>(buf.ptr);
    std::vector<double> tgt(data, data + handle_ids.size() * 3);

    DeformMethod method = parse_method(method_str);

    // Run C++ backend
    auto [verts_flat, faces_flat, meta] = deform_surface(
        mesh_path, handle_ids, tgt, roi_ids, method, alpha, max_iter);

    // Convert flat verts → numpy (N, 3)
    int nv = (int)verts_flat.size() / 3;
    py::array_t<double> V_new({nv, 3});
    std::copy(verts_flat.begin(), verts_flat.end(),
              V_new.mutable_data());

    // Convert flat faces → numpy (M, 3)
    int nf = (int)faces_flat.size() / 3;
    py::array_t<int> F({nf, 3});
    std::copy(faces_flat.begin(), faces_flat.end(),
              F.mutable_data());

    // Build metadata dict
    py::dict meta_dict;
    meta_dict["template_mesh_path"] = meta.template_mesh_path;
    meta_dict["method"]             = meta.method;
    meta_dict["handle_ids"]         = meta.handle_ids;
    meta_dict["target_positions"]   = meta.target_positions;
    meta_dict["roi_ids"]            = meta.roi_ids;
    meta_dict["alpha"]              = meta.alpha;
    meta_dict["max_iter"]           = meta.max_iter;

    return py::make_tuple(V_new, F, meta_dict);
}

// ── Module definition ─────────────────────────────────────────────────────────

PYBIND11_MODULE(graphop, m)
{
    m.doc() = R"doc(
graphop — CGAL-based surface deformation backend.

Functions
---------
deform_surface(mesh_path, handle_ids, target_positions, ...)
    Deform a triangulated surface mesh using ARAP or SRE-ARAP.
)doc";

    m.def(
        "deform_surface",
        &py_deform_surface,
        R"doc(
Deform a triangulated surface mesh loaded from an OBJ file.

Parameters
----------
mesh_path : str
    Path to the input OBJ mesh file.
handle_ids : list[int]
    0-based vertex indices used as positional handles (control points).
target_positions : np.ndarray, shape (H, 3) or (3*H,)
    Target 3-D positions for each handle vertex.
roi_ids : list[int], optional
    Region-of-interest vertex indices.  Empty list (default) = whole mesh.
method : str, optional
    Deformation algorithm.  One of 'sre_arap' (default), 'original_arap',
    'spokes_and_rims'.
alpha : float, optional
    SRE-ARAP smoothness weight (default 0.02; ignored for other methods).
max_iter : int, optional
    Maximum ARAP iterations (default 50).

Returns
-------
V_new : np.ndarray, shape (N, 3), dtype float64
    Deformed vertex positions.
F : np.ndarray, shape (M, 3), dtype int32
    Face connectivity (0-based indices, unchanged from input mesh).
meta : dict
    Deformation metadata: template_mesh_path, method, handle_ids,
    target_positions, roi_ids, alpha, max_iter.
)doc",
        py::arg("mesh_path"),
        py::arg("handle_ids"),
        py::arg("target_positions"),
        py::arg("roi_ids")    = std::vector<int>{},
        py::arg("method")     = std::string("sre_arap"),
        py::arg("alpha")      = 0.02,
        py::arg("max_iter")   = 50);
}
