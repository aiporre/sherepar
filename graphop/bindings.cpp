/**
 * bindings.cpp
 *
 * pybind11 bindings for the graphop deformation backend.
 *
 * Exposes two functions to Python:
 *
 *   deform_surface(mesh_path, handle_ids, target_positions, ...)
 *       Classic interface: caller supplies raw target positions.
 *
 *   deform_surface_with_angles(mesh_path, handle_transforms, ...)
 *       Rotation interface: each handle specifies a center vertex, an angle,
 *       and a ring_size.  Target positions are computed automatically.
 *
 *       handle_transforms is a list of dicts, each with keys:
 *           vertex_id  (int)    — 0-based center vertex index
 *           angle      (float)  — rotation angle in radians
 *           ring_size  (float)  — radius; all vertices within this distance
 *                                 of the center vertex become handles
 *           center_coords (list[float], optional len=3) — custom rotation center
 *
 * Both functions return (V_new, F, meta) where:
 *   V_new — NumPy array [N, 3] float64
 *   F     — NumPy array [M, 3] int32
 *   meta  — Python dict with all deformation metadata
 */

#include "deformation.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace graphop;

// ── Shared helpers ────────────────────────────────────────────────────────────

static DeformMethod parse_method(const std::string& name)
{
    if (name == "sre_arap")         return DeformMethod::SRE_ARAP;
    if (name == "original_arap")    return DeformMethod::ORIGINAL_ARAP;
    if (name == "spokes_and_rims")  return DeformMethod::SPOKES_AND_RIMS;
    throw std::invalid_argument(
        "Unknown method '" + name + "'. Choose from: "
        "'sre_arap', 'original_arap', 'spokes_and_rims'.");
}

/** Convert (verts_flat, faces_flat, meta) → Python (V_new, F, meta_dict). */
static py::tuple pack_result(
    const std::vector<double>& verts_flat,
    const std::vector<int>&    faces_flat,
    const DeformMeta&          meta)
{
    int nv = (int)verts_flat.size() / 3;
    py::array_t<double> V_new({nv, 3});
    std::copy(verts_flat.begin(), verts_flat.end(), V_new.mutable_data());

    int nf = (int)faces_flat.size() / 3;
    py::array_t<int> F({nf, 3});
    std::copy(faces_flat.begin(), faces_flat.end(), F.mutable_data());

    py::dict d;
    d["template_mesh_path"]      = meta.template_mesh_path;
    d["method"]                  = meta.method;
    d["handle_ids"]              = meta.handle_ids;
    d["target_positions"]        = meta.target_positions;
    d["roi_ids"]                 = meta.roi_ids;
    d["alpha"]                   = meta.alpha;
    d["max_iter"]                = meta.max_iter;
    d["transform_center_ids"]    = meta.transform_center_ids;
    d["transform_angles"]        = meta.transform_angles;
    d["transform_ring_sizes"]    = meta.transform_ring_sizes;
    d["transform_center_coords"] = meta.transform_center_coords;

    return py::make_tuple(V_new, F, d);
}

// ── deform_surface ────────────────────────────────────────────────────────────

static py::tuple py_deform_surface(
    const std::string&       mesh_path,
    const std::vector<int>&  handle_ids,
    py::array_t<double>      target_positions_arr,
    const std::vector<int>&  roi_ids,
    const std::string&       method_str,
    double alpha,
    int    max_iter)
{
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

    auto [vf, ff, meta] = deform_surface(
        mesh_path, handle_ids, tgt, roi_ids, parse_method(method_str), alpha, max_iter);
    return pack_result(vf, ff, meta);
}

// ── deform_surface_with_angles ────────────────────────────────────────────────

static py::tuple py_deform_surface_with_angles(
    const std::string&      mesh_path,
    const py::list&         handle_transforms_list,
    const std::vector<int>& roi_ids,
    const std::string&      method_str,
    double alpha,
    int    max_iter)
{
    std::vector<HandleTransform> transforms;
    transforms.reserve(handle_transforms_list.size());

    for (auto item : handle_transforms_list) {
        py::dict d = item.cast<py::dict>();

        if (!d.contains("vertex_id"))
            throw std::invalid_argument("Each handle transform dict must have 'vertex_id'");
        if (!d.contains("angle"))
            throw std::invalid_argument("Each handle transform dict must have 'angle'");
        if (!d.contains("ring_size"))
            throw std::invalid_argument("Each handle transform dict must have 'ring_size'");

        HandleTransform t;
        t.vertex_id = d["vertex_id"].cast<int>();
        t.angle     = d["angle"].cast<double>();
        t.ring_size = d["ring_size"].cast<double>();

        if (d.contains("center_coords")) {
            t.center_coords = d["center_coords"].cast<std::vector<double>>();
            if (t.center_coords.size() != 3) {
                throw std::invalid_argument(
                    "'center_coords' must be a 3-element list [x, y, z] when provided");
            }
        }

        transforms.push_back(t);
    }

    auto [vf, ff, meta] = deform_surface_with_angles(
        mesh_path, transforms, roi_ids, parse_method(method_str), alpha, max_iter);
    return pack_result(vf, ff, meta);
}

// ── Module definition ─────────────────────────────────────────────────────────

PYBIND11_MODULE(graphop, m)
{
    m.doc() = R"doc(
graphop — CGAL-based surface deformation backend for pmConv Stage 1.

Functions
---------
deform_surface(mesh_path, handle_ids, target_positions, ...)
    Classic interface: deform using explicit target positions per handle.

deform_surface_with_angles(mesh_path, handle_transforms, ...)
    Rotation interface: each handle specifies vertex_id, angle, ring_size.
)doc";

    m.def(
        "deform_surface",
        &py_deform_surface,
        py::arg("mesh_path"),
        py::arg("handle_ids"),
        py::arg("target_positions"),
        py::arg("roi_ids")  = std::vector<int>{},
        py::arg("method")   = std::string("sre_arap"),
        py::arg("alpha")    = 0.02,
        py::arg("max_iter") = 50);

    m.def(
        "deform_surface_with_angles",
        &py_deform_surface_with_angles,
        py::arg("mesh_path"),
        py::arg("handle_transforms"),
        py::arg("roi_ids")  = std::vector<int>{},
        py::arg("method")   = std::string("sre_arap"),
        py::arg("alpha")    = 0.02,
        py::arg("max_iter") = 50);
}