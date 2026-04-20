/**
 * test_deformation.cpp
 *
 * C++ test battery for the graphop deformation backend.
 *
 * Covers:
 *  - deform_surface()             (error handling + three algorithms)
 *  - deform_surface_with_angles() (ring selection, rotation, error handling)
 *  - rodrigues_rotate geometry    (via deform_surface_with_angles results)
 *
 * Build and run (from the repo root):
 *
 *   mkdir -p build && cd build
 *   cmake .. -DCMAKE_BUILD_TYPE=Release
 *   make test_graphop
 *   ./test_graphop
 *   # or: ctest --test-dir . -V
 *
 * The TEST_DATA_DIR macro is defined by CMake to point at <repo>/data/.
 * Add new void test_xxx() functions and a RUN_TEST(test_xxx) call in main()
 * to extend the suite.
 */

#include "test_framework.h"
#include "deformation.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

#ifndef TEST_DATA_DIR
#  error "TEST_DATA_DIR must be defined via CMake (see CMakeLists.txt)"
#endif

static const std::string ELLIPSOID_OBJ = std::string(TEST_DATA_DIR) + "/ellipsoid.obj";

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Max absolute difference between two same-size flat vertex arrays. */
static double max_vertex_diff(const std::vector<double>& a,
                               const std::vector<double>& b)
{
    double mx = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        mx = std::max(mx, std::abs(a[i] - b[i]));
    return mx;
}

/** Count vertices within ring_size of center_id in the original mesh. */
static int count_vertices_in_ring(const std::vector<double>& verts,
                                   int center_id,
                                   double ring_size)
{
    double cx = verts[3*center_id], cy = verts[3*center_id+1], cz = verts[3*center_id+2];
    int count = 0;
    int nv = (int)verts.size() / 3;
    for (int i = 0; i < nv; ++i) {
        double dx = verts[3*i]-cx, dy = verts[3*i+1]-cy, dz = verts[3*i+2]-cz;
        if (dx*dx + dy*dy + dz*dz <= ring_size*ring_size)
            ++count;
    }
    return count;
}

/** Quick-and-dirty OBJ vertex reader (matches the one in deformation.cpp). */
static std::vector<double> read_obj_vertices(const std::string& path)
{
    std::ifstream in(path);
    std::vector<double> verts;
    std::string line;
    while (std::getline(in, line)) {
        if (line.size() < 2 || line[0] != 'v' || line[1] != ' ') continue;
        std::istringstream ss(line.substr(2));
        double x, y, z; ss >> x >> y >> z;
        verts.push_back(x); verts.push_back(y); verts.push_back(z);
    }
    return verts;
}

static std::vector<int> read_obj_faces(const std::string& path)
{
    std::ifstream in(path);
    std::vector<int> faces;
    std::string line;
    while (std::getline(in, line)) {
        if (line.size() < 2 || line[0] != 'f' || line[1] != ' ') continue;
        std::istringstream ss(line.substr(2));
        std::string tok; std::vector<int> idx;
        while (ss >> tok)
            idx.push_back(std::stoi(tok.substr(0, tok.find('/'))) - 1);
        for (int k = 1; k + 1 < (int)idx.size(); ++k) {
            faces.push_back(idx[0]);
            faces.push_back(idx[k]);
            faces.push_back(idx[k+1]);
        }
    }
    return faces;
}

// ── deform_surface — error handling ──────────────────────────────────────────

void test_error_bad_file()
{
    EXPECT_THROWS(
        graphop::deform_surface("/no/such/file.obj", {0}, {0,0,0})
    );
}

void test_error_handle_id_out_of_range()
{
    // mesh has ~400 vertices; 99999 is way out of range
    EXPECT_THROWS(
        graphop::deform_surface(ELLIPSOID_OBJ, {99999}, {0,0,0})
    );
}

void test_error_negative_handle_id()
{
    EXPECT_THROWS(
        graphop::deform_surface(ELLIPSOID_OBJ, {-1}, {0,0,0})
    );
}

void test_error_target_size_mismatch()
{
    // 2 handles but only 3 target values (need 6)
    EXPECT_THROWS(
        graphop::deform_surface(ELLIPSOID_OBJ, {0, 1}, {0, 0, 0})
    );
}

void test_error_empty_handles()
{
    // 0 handle_ids, 0 targets — CGAL should fail preprocessing
    EXPECT_THROWS(
        graphop::deform_surface(ELLIPSOID_OBJ, {}, {})
    );
}

// ── deform_surface — correctness ─────────────────────────────────────────────

void test_sre_arap_vertex_count()
{
    auto orig = read_obj_vertices(ELLIPSOID_OBJ);
    int nv_orig = (int)orig.size() / 3;

    auto [vf, ff, meta] = graphop::deform_surface(
        ELLIPSOID_OBJ, {0, 50, 100}, {0,0,0, 1,0,0, 0,1,0},
        {}, graphop::DeformMethod::SRE_ARAP);

    EXPECT_EQ((int)vf.size() / 3, nv_orig);
    EXPECT_EQ(meta.method, std::string("sre_arap"));
}

void test_sre_arap_face_count()
{
    // Face count must be identical between two different deformations of the
    // same mesh — deformation never changes topology.
    // (CGAL may load fewer faces than the raw OBJ due to manifold validation;
    //  we compare deformed vs identity-deformed to isolate the topology check.)
    auto [vf_ref, ff_ref, m_ref] = graphop::deform_surface(
        ELLIPSOID_OBJ, {0, 50}, {0,0,0, 1,0,0},
        {}, graphop::DeformMethod::SRE_ARAP);

    auto [vf, ff, meta] = graphop::deform_surface(
        ELLIPSOID_OBJ, {0, 50, 100}, {0.1,0,0, 1,0.1,0, 0,1,0.1},
        {}, graphop::DeformMethod::SRE_ARAP);

    EXPECT_EQ((int)ff.size() / 3, (int)ff_ref.size() / 3);
}

void test_original_arap_runs()
{
    auto [vf, ff, meta] = graphop::deform_surface(
        ELLIPSOID_OBJ, {0, 100}, {0,0,0, 1,0,0},
        {}, graphop::DeformMethod::ORIGINAL_ARAP);

    EXPECT_TRUE(vf.size() > 0);
    EXPECT_EQ(meta.method, std::string("original_arap"));
}

void test_spokes_and_rims_runs()
{
    auto [vf, ff, meta] = graphop::deform_surface(
        ELLIPSOID_OBJ, {0, 100}, {0,0,0, 1,0,0},
        {}, graphop::DeformMethod::SPOKES_AND_RIMS);

    EXPECT_TRUE(vf.size() > 0);
    EXPECT_EQ(meta.method, std::string("spokes_and_rims"));
}

void test_meta_handle_ids_stored()
{
    std::vector<int> hids = {0, 50, 100};
    auto [vf, ff, meta] = graphop::deform_surface(
        ELLIPSOID_OBJ, hids, {0,0,0, 1,0,0, 0,1,0});

    EXPECT_EQ(meta.handle_ids, hids);
}

void test_zero_displacement_no_significant_change()
{
    // Handles at their original positions — mesh should barely move
    auto orig = read_obj_vertices(ELLIPSOID_OBJ);

    // Use vertex 0 and 50 at their original positions
    std::vector<int>    hids = {0, 50};
    std::vector<double> tgt  = {
        orig[0], orig[1], orig[2],
        orig[150], orig[151], orig[152]
    };

    auto [vf, ff, meta] = graphop::deform_surface(
        ELLIPSOID_OBJ, hids, tgt);

    double diff = max_vertex_diff(vf, orig);
    EXPECT_TRUE(diff < 0.1);  // negligible movement
}

// ── deform_surface_with_angles — error handling ───────────────────────────────

void test_with_angles_empty_transforms()
{
    EXPECT_THROWS(
        graphop::deform_surface_with_angles(ELLIPSOID_OBJ, {})
    );
}

void test_with_angles_bad_vertex_id()
{
    graphop::HandleTransform t{99999, 0.3, 0.5};
    EXPECT_THROWS(
        graphop::deform_surface_with_angles(ELLIPSOID_OBJ, {t})
    );
}

void test_with_angles_bad_file()
{
    graphop::HandleTransform t{0, 0.3, 0.5};
    EXPECT_THROWS(
        graphop::deform_surface_with_angles("/no/such/file.obj", {t})
    );
}

// ── deform_surface_with_angles — correctness ─────────────────────────────────

void test_with_angles_vertex_count_preserved()
{
    auto orig = read_obj_vertices(ELLIPSOID_OBJ);
    int nv = (int)orig.size() / 3;

    graphop::HandleTransform t{0, 0.2, 0.5};
    auto [vf, ff, meta] = graphop::deform_surface_with_angles(ELLIPSOID_OBJ, {t});

    EXPECT_EQ((int)vf.size() / 3, nv);
}

void test_with_angles_face_count_preserved()
{
    // Face count must be identical between two different deformations.
    graphop::HandleTransform t1{0, 0.0, 0.5};
    graphop::HandleTransform t2{0, 0.2, 0.5};
    auto [vf_ref, ff_ref, m_ref] = graphop::deform_surface_with_angles(ELLIPSOID_OBJ, {t1});
    auto [vf, ff, meta]          = graphop::deform_surface_with_angles(ELLIPSOID_OBJ, {t2});

    EXPECT_EQ((int)ff.size() / 3, (int)ff_ref.size() / 3);
}

void test_with_angles_meta_transform_fields_stored()
{
    graphop::HandleTransform t{10, 0.4, 0.3};
    auto [vf, ff, meta] = graphop::deform_surface_with_angles(ELLIPSOID_OBJ, {t});

    EXPECT_EQ((int)meta.transform_center_ids.size(), 1);
    EXPECT_EQ(meta.transform_center_ids[0], 10);
    EXPECT_NEAR(meta.transform_angles[0], 0.4, 1e-12);
    EXPECT_NEAR(meta.transform_ring_sizes[0], 0.3, 1e-12);
}

void test_with_angles_ring_covers_center_at_minimum()
{
    // ring_size = 0 → only the center vertex; handle_ids must contain vertex_id
    graphop::HandleTransform t{5, 0.2, 0.0};
    auto [vf, ff, meta] = graphop::deform_surface_with_angles(ELLIPSOID_OBJ, {t});

    // Center vertex must be in the expanded handle list
    EXPECT_TRUE(std::find(meta.handle_ids.begin(),
                          meta.handle_ids.end(), 5) != meta.handle_ids.end());
}

void test_with_angles_ring_size_selects_more_vertices()
{
    auto orig = read_obj_vertices(ELLIPSOID_OBJ);

    double small_ring = 0.05;
    double large_ring = 0.50;

    graphop::HandleTransform t_small{0, 0.1, small_ring};
    graphop::HandleTransform t_large{0, 0.1, large_ring};

    auto [vf_s, ff_s, meta_s] = graphop::deform_surface_with_angles(ELLIPSOID_OBJ, {t_small});
    auto [vf_l, ff_l, meta_l] = graphop::deform_surface_with_angles(ELLIPSOID_OBJ, {t_large});

    // Larger ring → more handles
    EXPECT_TRUE(meta_l.handle_ids.size() > meta_s.handle_ids.size());
}

void test_with_angles_zero_angle_minimal_change()
{
    // angle = 0 → Rodrigues rotation is identity → handles stay at original
    // positions → mesh should barely deform (similar to zero_displacement test)
    auto orig = read_obj_vertices(ELLIPSOID_OBJ);

    graphop::HandleTransform t{0, 0.0, 0.3};
    auto [vf, ff, meta] = graphop::deform_surface_with_angles(ELLIPSOID_OBJ, {t});

    double diff = max_vertex_diff(vf, orig);
    EXPECT_TRUE(diff < 0.1);
}

void test_with_angles_nonzero_angle_changes_mesh()
{
    auto orig = read_obj_vertices(ELLIPSOID_OBJ);

    // A meaningful rotation on a large ring should move the mesh noticeably
    graphop::HandleTransform t{0, 0.5, 0.8};
    auto [vf, ff, meta] = graphop::deform_surface_with_angles(ELLIPSOID_OBJ, {t});

    double diff = max_vertex_diff(vf, orig);
    EXPECT_TRUE(diff > 1e-6);  // mesh should have moved
}

void test_with_angles_multiple_transforms()
{
    graphop::HandleTransform t1{0,   0.2, 0.3};
    graphop::HandleTransform t2{100, 0.3, 0.3};

    // Should complete without error and produce correct topology
    auto orig = read_obj_vertices(ELLIPSOID_OBJ);
    auto [vf, ff, meta] = graphop::deform_surface_with_angles(ELLIPSOID_OBJ, {t1, t2});

    EXPECT_EQ((int)vf.size() / 3, (int)orig.size() / 3);
    EXPECT_EQ((int)meta.transform_center_ids.size(), 2);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main()
{
    std::cout << "\n=== graphop deformation test battery ===\n\n";

    std::cout << "--- deform_surface: error handling ---\n";
    RUN_TEST(test_error_bad_file);
    RUN_TEST(test_error_handle_id_out_of_range);
    RUN_TEST(test_error_negative_handle_id);
    RUN_TEST(test_error_target_size_mismatch);
    RUN_TEST(test_error_empty_handles);

    std::cout << "\n--- deform_surface: correctness ---\n";
    RUN_TEST(test_sre_arap_vertex_count);
    RUN_TEST(test_sre_arap_face_count);
    RUN_TEST(test_original_arap_runs);
    RUN_TEST(test_spokes_and_rims_runs);
    RUN_TEST(test_meta_handle_ids_stored);
    RUN_TEST(test_zero_displacement_no_significant_change);

    std::cout << "\n--- deform_surface_with_angles: error handling ---\n";
    RUN_TEST(test_with_angles_empty_transforms);
    RUN_TEST(test_with_angles_bad_vertex_id);
    RUN_TEST(test_with_angles_bad_file);

    std::cout << "\n--- deform_surface_with_angles: correctness ---\n";
    RUN_TEST(test_with_angles_vertex_count_preserved);
    RUN_TEST(test_with_angles_face_count_preserved);
    RUN_TEST(test_with_angles_meta_transform_fields_stored);
    RUN_TEST(test_with_angles_ring_covers_center_at_minimum);
    RUN_TEST(test_with_angles_ring_size_selects_more_vertices);
    RUN_TEST(test_with_angles_zero_angle_minimal_change);
    RUN_TEST(test_with_angles_nonzero_angle_changes_mesh);
    RUN_TEST(test_with_angles_multiple_transforms);

    return test::summary();
}