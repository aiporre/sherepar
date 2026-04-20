/**
 * deformation.cpp
 *
 * CGAL Surface_mesh_deformation backend for pmConv PoF.
 *
 * Reads an OBJ mesh, applies ARAP / SRE-ARAP deformation with
 * positional constraints, and returns the deformed
 * > vertex positions
 * > faces
 * > metadata about the deformation (input parameters, method used, etc.)
 *
 */

#include "deformation.h"

// CGAL kernel + mesh
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Surface_mesh_deformation.h>

// I/O to read stuff
#include <CGAL/IO/OBJ.h>

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <map>

// Aliases:  short names
using Kernel    = CGAL::Simple_cartesian<double>;
using Point_3   = Kernel::Point_3;
using SurfMesh  = CGAL::Surface_mesh<Point_3>;
using VD        = SurfMesh::Vertex_index;
using vertex_iterator = SurfMesh::Vertex_iterator;
using edge_descriptor = SurfMesh::edge_descriptor;
using halfedge_descriptor = SurfMesh::halfedge_descriptor;
using vertex_descriptor = SurfMesh::vertex_descriptor;
using vertex_ring = std::vector<vertex_descriptor>;
// Deformation object instantiated for SRE_ARAP (the richest; we use the same
// type and just ignore alpha when running plain ARAP).
// This another shortcut to avoid writting the full template signature everywhere.
// The actual algorithm is selected at runtime via the DeformMethod enum,
// but we need to instantiate the template for all cases
template <CGAL::Deformation_algorithm_tag TAG>
using Deformer = CGAL::Surface_mesh_deformation<SurfMesh, CGAL::Default,
                                                 CGAL::Default, TAG>;

// here we implement helpers

/**
 * OBJ file reader helper: reads vertices and faces from a minimal OBJ file.
 * Parse a minimal OBJ file: only 'v' and 'f' lines, triangulated faces only.
 * Returns true on success.
 */
static bool load_obj(const std::string& path,
                     std::vector<double>& verts,
                     std::vector<int>& faces)
{
    std::ifstream in(path);
    if (!in)
        return false;

    std::string line;
    while (std::getline(in, line)) {
        if (line.size() < 2) continue;
        if (line[0] == 'v' && line[1] == ' ') {
            std::istringstream ss(line.substr(2));
            double x, y, z;
            ss >> x >> y >> z;
            verts.push_back(x); verts.push_back(y); verts.push_back(z);
        } else if (line[0] == 'f' && line[1] == ' ') {
            std::istringstream ss(line.substr(2));
            std::string tok;
            std::vector<int> idx;
            while (ss >> tok) {
                // handle "v", "v/vt", "v/vt/vn"
                int v = std::stoi(tok.substr(0, tok.find('/')));
                idx.push_back(v - 1); // OBJ is 1-based
            }
            // triangulate fans for polygons
            for (int k = 1; k + 1 < (int)idx.size(); ++k) {
                faces.push_back(idx[0]);
                faces.push_back(idx[k]);
                faces.push_back(idx[k + 1]);
            }
        }
    }
    return !verts.empty() && !faces.empty();
}

/**
 * Build a CGAL Surface_mesh from flat vertex / face arrays.
 */
static SurfMesh build_cgal_mesh(const std::vector<double>& verts,
                                 const std::vector<int>& faces)
{
    SurfMesh mesh;
    int nv = (int)verts.size() / 3;
    mesh.reserve(nv, 0, (int)faces.size() / 3);

    for (int i = 0; i < nv; ++i)
        mesh.add_vertex(Point_3(verts[3*i], verts[3*i+1], verts[3*i+2]));

    int nf = (int)faces.size() / 3;
    for (int i = 0; i < nf; ++i)
        mesh.add_face(VD(faces[3*i]), VD(faces[3*i+1]), VD(faces[3*i+2]));

    return mesh;
}

/**
 * Extract deformed vertices from the mesh into a flat double array.
 */
static std::vector<double> extract_vertices(const SurfMesh& mesh)
{
    std::vector<double> out;
    out.reserve(mesh.num_vertices() * 3);
    for (auto v : mesh.vertices()) {
        const auto& p = mesh.point(v);
        out.push_back(p.x()); out.push_back(p.y()); out.push_back(p.z());
    }
    return out;
}

/**
 * Extract face connectivity from the mesh into a flat int array.
 */
static std::vector<int> extract_faces(const SurfMesh& mesh)
{
    std::vector<int> out;
    out.reserve(mesh.num_faces() * 3);
    for (auto f : mesh.faces()) {
        auto h = mesh.halfedge(f);
        for (auto v : mesh.vertices_around_face(h))
            out.push_back(v.idx());
    }
    return out;
}

// names for the DeformMethod enum, used in the metadata

static std::string method_name(graphop::DeformMethod m)
{
    switch (m) {
        case graphop::DeformMethod::ORIGINAL_ARAP:   return "original_arap";
        case graphop::DeformMethod::SPOKES_AND_RIMS: return "spokes_and_rims";
        case graphop::DeformMethod::SRE_ARAP:        return "sre_arap";
    }
    return "unknown";
}

// extract a ring using iterators as in the example in CGAL docs
// Collect the vertices which are at distance less or equal to k
// from the vertex v in the graph of vertices connected by the edges of mesh
static vertex_ring extract_k_ring(const SurfMesh& mesh, vertex_descriptor v, int k)
{
  std::map<vertex_descriptor, int>  D; // collection of vertices and their distance from v
  vertex_ring                       Q; // vertices in set of near to v, initialized with v
  Q.push_back(v); D[v] = 0;
  std::size_t current_index = 0;

  while (current_index < Q.size()) {
    const vertex_descriptor current_v = Q[current_index];
    const int dist_v = D[current_v];
    if (dist_v >= k)
      break;

    ++current_index;

    for (vertex_descriptor new_v : mesh.vertices_around_target(mesh.halfedge(current_v))) {
      if (D.insert(std::make_pair(new_v, dist_v + 1)).second)
        Q.push_back(new_v);
    }
  }
  return Q;
}

// compute normal to a vertex
static Vector_3 vertex_normal(const SurfMesh& mesh, vertex_descriptor v)
{
    Vector_3 n(0.0, 0.0, 0.0);

    auto h = mesh.halfedge(v);
    if (h == SurfMesh::null_halfedge()) {
        return n;
    }

    for (auto f : CGAL::faces_around_target(h, mesh)) {
        if (f == SurfMesh::null_face())
            continue;

        n = n + CGAL::Polygon_mesh_processing::compute_face_normal(f, mesh);
    }

    const double len = std::sqrt(n.squared_length());
    if (len > 0.0)
        n = n / len;

    return n;
}


// run deformation for a specific CGAL::Deformation_algorithm_tag, called by the public API

template <CGAL::Deformation_algorithm_tag TAG>
static std::tuple<std::vector<double>, std::vector<int>, graphop::DeformMeta>
run_deformation(
    const std::string& mesh_path,
    const std::vector<int>& handle_ids,
    const std::vector<double>& target_positions,
    const std::vector<int>& roi_ids,
    double alpha,
    int max_iter)
{
    // load obj and build CGAL mesh
    std::vector<double> raw_verts;
    std::vector<int>    raw_faces;
    if (!load_obj(mesh_path, raw_verts, raw_faces))
        throw std::runtime_error("Failed to load OBJ file: " + mesh_path);

    SurfMesh mesh = build_cgal_mesh(raw_verts, raw_faces);


    int nv = (int)mesh.num_vertices();

    //  index validaiton
    std::cout << "handle_ids: " << handle_ids.size() << " handles, roi_ids: " << roi_ids.size() << " ROI vertices\n";
    for (int id : handle_ids)
        if (id < 0 || id >= nv)
            throw std::runtime_error("handle_id " + std::to_string(id) +
                                     " out of range [0, " + std::to_string(nv) + ")");

    if ((int)target_positions.size() != (int)handle_ids.size() * 3)
        throw std::runtime_error("target_positions length must equal 3 * handle_ids.size()");

    for (int id : roi_ids)
        if (id < 0 || id >= nv)
            throw std::runtime_error("roi_id " + std::to_string(id) +
                                     " out of range [0, " + std::to_string(nv) + ")");

    //  Deformation instance
    Deformer<TAG> deformer(mesh);

    if constexpr (TAG == CGAL::SRE_ARAP)
        deformer.set_sre_arap_alpha(alpha);

    // Region of interest: full mesh if roi_ids is empty
    if (roi_ids.empty()) {
        std::cout << "Using full mesh as ROI (" << nv << " vertices)\n";
        for (auto v : mesh.vertices())
            deformer.insert_roi_vertex(v);
    } else {
        std::cout << "Using specified ROI with " << roi_ids.size() << " vertices\n";
        for (int id : roi_ids)
            deformer.insert_roi_vertex(VD(id));
    }

    // Control vertices (handles)
    for (int id : handle_ids)
        deformer.insert_control_vertex(VD(id));

    bool ok = deformer.preprocess();
    if (!ok)
        throw std::runtime_error("CGAL deformer preprocessing failed; "
                                 "check mesh validity and handle/ROI configuration.");

    // Apply target positions
    for (int i = 0; i < (int)handle_ids.size(); ++i) {
        Point_3 tgt(target_positions[3*i],
                    target_positions[3*i+1],
                    target_positions[3*i+2]);
        deformer.set_target_position(VD(handle_ids[i]), tgt);
    }

    deformer.deform(static_cast<unsigned int>(max_iter), /*tolerance=*/1e-4);

    // ── Collect results ────────────────────────────────────────────────────
    auto out_verts = extract_vertices(mesh);
    auto out_faces = extract_faces(mesh);

    graphop::DeformMeta meta;
    meta.template_mesh_path = mesh_path;
    meta.method              = method_name(
        TAG == CGAL::ORIGINAL_ARAP   ? graphop::DeformMethod::ORIGINAL_ARAP :
        TAG == CGAL::SPOKES_AND_RIMS ? graphop::DeformMethod::SPOKES_AND_RIMS :
                                       graphop::DeformMethod::SRE_ARAP);
    meta.handle_ids          = handle_ids;
    meta.target_positions    = target_positions;
    meta.roi_ids             = roi_ids;
    meta.alpha               = alpha;
    meta.max_iter            = max_iter;

    return {out_verts, out_faces, meta};
}



template <CGAL::Deformation_algorithm_tag TAG>
static std::tuple<std::vector<double>, std::vector<int>, graphop::DeformMeta>
run_deformation_with_angle(
    const std::string& mesh_path,
    const std::vector<int>& handle_ids,
    const std::vector<double>& target_positions,
    const std::vector<int>& roi_ids,
    double angle,
    double alpha,
    int max_iter)
{
    // load obj and build CGAL mesh
    std::vector<double> raw_verts;
    std::vector<int>    raw_faces;
    if (!load_obj(mesh_path, raw_verts, raw_faces))
        throw std::runtime_error("Failed to load OBJ file: " + mesh_path);

    SurfMesh mesh = build_cgal_mesh(raw_verts, raw_faces);


    int nv = (int)mesh.num_vertices();

    //  index validaiton
    std::cout << "handle_ids: " << handle_ids.size() << " handles, roi_ids: " << roi_ids.size() << " ROI vertices\n";
    for (int id : handle_ids)
        if (id < 0 || id >= nv)
            throw std::runtime_error("handle_id " + std::to_string(id) +
                                     " out of range [0, " + std::to_string(nv) + ")");

    if ((int)target_positions.size() != (int)handle_ids.size() * 3)
        throw std::runtime_error("target_positions length must equal 3 * handle_ids.size()");

    for (int id : roi_ids)
        if (id < 0 || id >= nv)
            throw std::runtime_error("roi_id " + std::to_string(id) +
                                     " out of range [0, " + std::to_string(nv) + ")");

    //  Deformation instance
    Deformer<TAG> deformer(mesh);

    if constexpr (TAG == CGAL::SRE_ARAP)
        deformer.set_sre_arap_alpha(alpha);

    // Region of interest: full mesh if roi_ids is empty
    if (roi_ids.empty()) {
        std::cout << "Using full mesh as ROI (" << nv << " vertices)\n";
//        vertex_iterator vb, ve;
//        std::tie(vb,ve) = vertices(mesh);
//        deformer.insert_roi_vertices(vb, ve);
        for (auto v : mesh.vertices())
            deformer.insert_roi_vertex(v);
    } else {
        std::cout << "Using specified ROI with " << roi_ids.size() << " vertices\n";
        for (int id : roi_ids)
            deformer.insert_roi_vertex(VD(id));
    }

    // list of vertices descriptors for the rings
    std::vector<vertex_ring> rings;
    // Control vertices (handles)
    std::cout << "Using rotation and translation constraints for handles\n";
    // deformer.set_rotation_and_translation_constraints(true);
    for (int id : handle_ids) {
        const auto ring = extract_k_ring(mesh, VD(id), 1);
        rings.push_back(ring);
        for (auto v : ring)
            deformer.insert_control_vertex(v);
        }
    }

    bool ok = deformer.preprocess();
    if (!ok)
        throw std::runtime_error("CGAL deformer preprocessing failed; "
                                 "check mesh validity and handle/ROI configuration.");

    // Apply target positions
    for (int i = 0; i < (int)handle_ids.size(); ++i) {
        const VD h = VD(handle_ids[i]);
        const vertex_ring& ring = rings[i];

        // translation target (as you already do)
        Point_3 translation(target_positions[3*i],
                            target_positions[3*i+1],
                            target_positions[3*i+2]);

        // axis = handle normal
        Vector_3 n = vertex_normal(mesh, h);
        const double nlen = std::sqrt(n.squared_length());
        if (nlen == 0.0) {
            throw std::runtime_error("Zero normal at handle " + std::to_string(handle_ids[i]));
        }

        Eigen::Vector3d axis(n.x() / nlen, n.y() / nlen, n.z() / nlen);

        // angle is in radians
        // TODO: use btter one angle per target for now it is the same for all targets
        Eigen::Quaterniond quat(Eigen::AngleAxisd(angle, axis));

        // CGAL API expects (control_vertex, target_position, quaternion)
        deformer.rotate(ring.begin(), ring.end(), translation, quat);

     }


    deformer.deform(static_cast<unsigned int>(max_iter), /*tolerance=*/1e-4);

    // ── Collect results ────────────────────────────────────────────────────
    auto out_verts = extract_vertices(mesh);
    auto out_faces = extract_faces(mesh);

    graphop::DeformMeta meta;
    meta.template_mesh_path = mesh_path;
    meta.method              = method_name(
        TAG == CGAL::ORIGINAL_ARAP   ? graphop::DeformMethod::ORIGINAL_ARAP :
        TAG == CGAL::SPOKES_AND_RIMS ? graphop::DeformMethod::SPOKES_AND_RIMS :
                                       graphop::DeformMethod::SRE_ARAP);
    meta.handle_ids          = handle_ids;
    meta.target_positions    = target_positions;
    meta.roi_ids             = roi_ids;
    meta.alpha               = alpha;
    meta.max_iter            = max_iter;

    return {out_verts, out_faces, meta};
}

// API exposed

namespace graphop {

std::tuple<std::vector<double>, std::vector<int>, DeformMeta>
deform_surface(
    const std::string& mesh_path,
    const std::vector<int>& handle_ids,
    const std::vector<double>& target_positions,
    const std::vector<int>& roi_ids,
    DeformMethod method,
    double alpha,
    int max_iter,
    bool use_rotation_and_translation)
{
    switch (method) {
        case DeformMethod::ORIGINAL_ARAP:
            return run_deformation<CGAL::ORIGINAL_ARAP>(
                mesh_path, handle_ids, target_positions, roi_ids, alpha, max_iter, use_rotation_and_translation);
        case DeformMethod::SPOKES_AND_RIMS:
            return run_deformation<CGAL::SPOKES_AND_RIMS>(
                mesh_path, handle_ids, target_positions, roi_ids, alpha, max_iter, use_rotation_and_translation);
        case DeformMethod::SRE_ARAP:
            return run_deformation<CGAL::SRE_ARAP>(
                mesh_path, handle_ids, target_positions, roi_ids, alpha, max_iter, use_rotation_and_translation);
    }
    throw std::runtime_error("Unknown deformation method");
}

} // namespace graphop