/**
 * deformation.cpp
 *
 * CGAL Surface_mesh_deformation backend for pmConv Stage 1.
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
#include <CGAL/Polygon_mesh_processing/compute_normal.h>

// I/O to read stuff
#include <CGAL/IO/OBJ.h>

// Eigen for quaternion-based rotations
#include <Eigen/Geometry>

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <map>

// Aliases: short names
using Kernel    = CGAL::Simple_cartesian<double>;
using Point_3   = Kernel::Point_3;
using Vector_3  = Kernel::Vector_3;
using SurfMesh  = CGAL::Surface_mesh<Point_3>;
using VD        = SurfMesh::Vertex_index;
// In CGAL 5.x the nested graph-traits types (vertex_descriptor, edge_descriptor,
// halfedge_descriptor) are NOT nested members of Surface_mesh — they live inside
// boost::graph_traits<SurfMesh>.  We keep the master branch's naming convention
// but bind them to the correct types via Vertex_index / Halfedge_index.
using vertex_descriptor    = VD;                            // = SurfMesh::Vertex_index
using vertex_ring          = std::vector<vertex_descriptor>;

// Deformation object instantiated for each algorithm tag.
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

/**
 * Collect all mesh vertices within Euclidean distance ring_size of center.
 * At minimum the center vertex itself is always included.
 */
static vertex_ring extract_euclidean_ring(const SurfMesh& mesh,
                                           vertex_descriptor center,
                                           double ring_size)
{
    const Point_3& cp = mesh.point(center);
    const double ring_sq = ring_size * ring_size;
    vertex_ring ring;
    for (auto v : mesh.vertices()) {
        const Point_3& p = mesh.point(v);
        double dx = p.x()-cp.x(), dy = p.y()-cp.y(), dz = p.z()-cp.z();
        if (dx*dx + dy*dy + dz*dz <= ring_sq)
            ring.push_back(v);
    }
    // guarantee center is present even when ring_size == 0
    if (ring.empty())
        ring.push_back(center);
    return ring;
}

// compute area-weighted normal at a vertex using CGAL's face normal helper
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

    //  index validation
    if (handle_ids.empty())
        throw std::runtime_error("handle_ids must not be empty");

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

    // Collect results
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


/**
 * Per-handle rotation deformation.
 *
 * For each handle:
 *   - collect all vertices within ring_size (Euclidean) of the center vertex
 *   - compute the surface normal at the center via area-weighted face normals
 *   - build an Eigen quaternion from AngleAxisd(angle, normal)
 *   - for each ring vertex v: target = center + quat * (v - center)
 *   - register all ring vertices as control vertices and set their targets
 */
template <CGAL::Deformation_algorithm_tag TAG>
static std::tuple<std::vector<double>, std::vector<int>, graphop::DeformMeta>
run_deformation_with_angle(
    const std::string& mesh_path,
    const std::vector<int>& handle_ids,
    const std::vector<double>& angles,
    const std::vector<double>& center_coords, // flat [x,y,z,...], one triple per handle
    const std::vector<char>& has_center_coords, // 1 if center_coords for handle is explicitly provided
    const std::vector<double>& ring_sizes,
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

    //  index validation
    if (handle_ids.empty())
        throw std::runtime_error("handle_ids must not be empty");

    std::cout << "handle_ids: " << handle_ids.size() << " handles, roi_ids: " << roi_ids.size() << " ROI vertices\n";
    for (int id : handle_ids)
        if (id < 0 || id >= nv)
            throw std::runtime_error("handle_id " + std::to_string(id) +
                                     " out of range [0, " + std::to_string(nv) + ")");

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

    // list of vertex rings, one per handle
    std::vector<vertex_ring> rings;
    std::cout << "Using rotation constraints for handles\n";
    for (int i = 0; i < (int)handle_ids.size(); ++i) {
        const auto ring = extract_euclidean_ring(mesh, VD(handle_ids[i]), ring_sizes[i]);
        rings.push_back(ring);
        for (auto v : ring)
            deformer.insert_control_vertex(v);
    }

    if (angles.size() != handle_ids.size())
        throw std::runtime_error("angles length must equal handle_ids.size()");
    if (ring_sizes.size() != handle_ids.size())
        throw std::runtime_error("ring_sizes length must equal handle_ids.size()");
    if (center_coords.size() != handle_ids.size() * 3)
        throw std::runtime_error("center_coords length must equal 3 * handle_ids.size()");
    if (has_center_coords.size() != handle_ids.size())
        throw std::runtime_error("has_center_coords length must equal handle_ids.size()");

    std::vector<Eigen::Vector3d> centers;
    centers.reserve(handle_ids.size());
    for (int i = 0; i < (int)handle_ids.size(); ++i) {
        if (has_center_coords[i]) {
            centers.emplace_back(center_coords[3*i], center_coords[3*i+1], center_coords[3*i+2]);
        } else {
            const Point_3& p = mesh.point(VD(handle_ids[i]));
            // centers.emplace_back(p.x(), p.y(), p.z());
            // put zeros
            centers.emplace_back(0.0, 0.0, 0.0);
        }
    }

    bool ok = deformer.preprocess();
    if (!ok)
        throw std::runtime_error("CGAL deformer preprocessing failed; "
                                 "check mesh validity and handle/ROI configuration.");

    // Apply per-handle rotation targets
    for (int i = 0; i < (int)handle_ids.size(); ++i) {
        const VD h = VD(handle_ids[i]);
//        const Point_3& center = mesh.point(h);
//        const Eigen::Vector3d center = Eigen::Vector3d(0.0, 0.0, 0.0); // rotation center is the origin, we rotate the displacements
        const Eigen::Vector3d center = centers[i];
        const vertex_ring& ring = rings[i];

        // axis = surface normal at handle center
        Vector_3 n = vertex_normal(mesh, h);
        const double nlen = std::sqrt(n.squared_length());
        if (nlen == 0.0)
            throw std::runtime_error("Zero normal at handle " + std::to_string(handle_ids[i]));

        Eigen::Vector3d axis(n.x() / nlen, n.y() / nlen, n.z() / nlen);

        // Build quaternion from angle-axis (Eigen)
        Eigen::Quaterniond quat(Eigen::AngleAxisd(angles[i], axis));

        // For each vertex in the ring: rotate its displacement around the center
        std::cout  << "we are rotating handle " << handle_ids[i] << " with angle " << angles[i]
                   << " and ring size " << ring_sizes[i] << ", affecting " << ring.size() << " vertices\n";
        deformer.rotate(ring.begin(), ring.end(), center, quat);
//        for (auto v : ring) {
//            const Point_3& p = mesh.point(v);
//            Eigen::Vector3d disp(p.x() - center.x(),
//                                 p.y() - center.y(),
//                                 p.z() - center.z());
//            Eigen::Vector3d rotated = quat * disp;
//            Point_3 target(center.x() + rotated.x(),
//                           center.y() + rotated.y(),
//                           center.z() + rotated.z());
//            deformer.set_target_position(v, target);
//        }
    }

    deformer.deform(static_cast<unsigned int>(max_iter), /*tolerance=*/1e-4);

    // Collect results
    auto out_verts = extract_vertices(mesh);
    auto out_faces = extract_faces(mesh);

    // Build metadata — flatten expanded handle ids + targets from the deformer state
    std::vector<int>    exp_handle_ids;
    std::vector<double> exp_targets;
    for (int i = 0; i < (int)handle_ids.size(); ++i) {
        for (auto v : rings[i]) {
            const Point_3& p = mesh.point(v); // deformed position
            exp_handle_ids.push_back(v.idx());
            exp_targets.push_back(p.x());
            exp_targets.push_back(p.y());
            exp_targets.push_back(p.z());
        }
    }

    graphop::DeformMeta meta;
    meta.template_mesh_path = mesh_path;
    meta.method              = method_name(
        TAG == CGAL::ORIGINAL_ARAP   ? graphop::DeformMethod::ORIGINAL_ARAP :
        TAG == CGAL::SPOKES_AND_RIMS ? graphop::DeformMethod::SPOKES_AND_RIMS :
                                       graphop::DeformMethod::SRE_ARAP);
    meta.handle_ids       = exp_handle_ids;
    meta.target_positions = exp_targets;
    meta.roi_ids          = roi_ids;
    meta.alpha            = alpha;
    meta.max_iter         = max_iter;
    meta.transform_center_coords.reserve(centers.size() * 3);
    for (const auto& c : centers) {
        meta.transform_center_coords.push_back(c.x());
        meta.transform_center_coords.push_back(c.y());
        meta.transform_center_coords.push_back(c.z());
    }

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
    int max_iter)
{
    switch (method) {
        case DeformMethod::ORIGINAL_ARAP:
            return run_deformation<CGAL::ORIGINAL_ARAP>(
                mesh_path, handle_ids, target_positions, roi_ids, alpha, max_iter);
        case DeformMethod::SPOKES_AND_RIMS:
            return run_deformation<CGAL::SPOKES_AND_RIMS>(
                mesh_path, handle_ids, target_positions, roi_ids, alpha, max_iter);
        case DeformMethod::SRE_ARAP:
            return run_deformation<CGAL::SRE_ARAP>(
                mesh_path, handle_ids, target_positions, roi_ids, alpha, max_iter);
    }
    throw std::runtime_error("Unknown deformation method");
}

std::tuple<std::vector<double>, std::vector<int>, DeformMeta>
deform_surface_with_angles(
    const std::string& mesh_path,
    const std::vector<HandleTransform>& handle_transforms,
    const std::vector<int>& roi_ids,
    DeformMethod method,
    double alpha,
    int max_iter)
{
    if (handle_transforms.empty())
        throw std::runtime_error("handle_transforms must not be empty");

    // unpack the per-handle parameters
    std::vector<int>    handle_ids;
    std::vector<double> angles, ring_sizes, center_coords;
    std::vector<char>   has_center_coords;
    center_coords.reserve(handle_transforms.size() * 3);
    has_center_coords.reserve(handle_transforms.size());
    for (const auto& t : handle_transforms) {
        handle_ids.push_back(t.vertex_id);
        angles.push_back(t.angle);
        ring_sizes.push_back(t.ring_size);

        if (!t.center_coords.empty()) {
            if (t.center_coords.size() != 3)
                throw std::runtime_error("center_coords for handle " + std::to_string(t.vertex_id) +
                                         " must have length 3 if provided");
            center_coords.push_back(t.center_coords[0]);
            center_coords.push_back(t.center_coords[1]);
            center_coords.push_back(t.center_coords[2]);
            has_center_coords.push_back(1);
        } else {
            center_coords.push_back(0.0);
            center_coords.push_back(0.0);
            center_coords.push_back(0.0);
            has_center_coords.push_back(0);
        }
    }

    std::tuple<std::vector<double>, std::vector<int>, DeformMeta> result;
    switch (method) {
        case DeformMethod::ORIGINAL_ARAP:
            result = run_deformation_with_angle<CGAL::ORIGINAL_ARAP>(
                mesh_path, handle_ids, angles, center_coords, has_center_coords, ring_sizes, roi_ids, alpha, max_iter);
            break;
        case DeformMethod::SPOKES_AND_RIMS:
            result = run_deformation_with_angle<CGAL::SPOKES_AND_RIMS>(
                mesh_path, handle_ids, angles, center_coords, has_center_coords, ring_sizes, roi_ids, alpha, max_iter);
            break;
        case DeformMethod::SRE_ARAP:
            result = run_deformation_with_angle<CGAL::SRE_ARAP>(
                mesh_path, handle_ids, angles, center_coords, has_center_coords, ring_sizes, roi_ids, alpha, max_iter);
            break;
        default:
            throw std::runtime_error("Unknown deformation method");
    }

    // augment metadata with the original transform parameters
    auto& meta = std::get<2>(result);
    for (const auto& t : handle_transforms) {
        meta.transform_center_ids.push_back(t.vertex_id);
        meta.transform_angles.push_back(t.angle);
        meta.transform_ring_sizes.push_back(t.ring_size);
    }

    return result;
}

} // namespace graphop