#include <stdint.h>

#include <iostream>
#include <unordered_set>

#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/stuff/sampler.h"
#include "g2o/types/sba/types_six_dof_expmap.h"

#if defined G2O_HAVE_CHOLMOD
G2O_USE_OPTIMIZATION_LIBRARY(cholmod);
#else
G2O_USE_OPTIMIZATION_LIBRARY(eigen);
#endif

G2O_USE_OPTIMIZATION_LIBRARY(dense);

using namespace Eigen;
using namespace std;

#define ROBUST_KERNEL false
#define STRUCTURE_ONLY false
#define DENSE false


class VisualSLAM3d {
    public:
        VisualSLAM3d(){
            optimizer_.setVerbose(false);
            string solverName = "lm_fix6_3";
            if (DENSE) {
                solverName = "lm_dense6_3";
            } else {
            #ifdef G2O_HAVE_CHOLMOD
                solverName = "lm_fix6_3_cholmod";
            #else
                solverName = "lm_fix6_3";
            #endif
            }
            g2o::OptimizationAlgorithmProperty solverProperty;
            optimizer_.setAlgorithm(
                g2o::OptimizationAlgorithmFactory::instance()->construct(solverName,
                                                               solverProperty));

            cout << "solver created" << endl;

        }

        bool setCameraParam(double focal_length, double c_x, double c_y){
            Vector2d principal_point(c_x, c_y);
            cam_params_ = new g2o::CameraParameters(focal_length, principal_point, 0.0);
            cam_params_->setId(0);
            if (!optimizer_.addParameter(cam_params_)) {
                return false;
            }

            cout << "cam param" << endl;
            return true;

        }

        Vector2d project(double x, double y, double z){
            Vector2d uv = cam_params_->cam_map(Vector3d(x, y, z));
            return uv;
        }

        /*
        bool addPoseVertex(int id, const Vector3d t, const g2o::Matrix3 R ){
            //Vector3d trans(x, y, z);
            g2o::SE3Quat pose(R, t);
            //pose.setTranslation(trans);
            g2o::VertexSE3Expmap* v_se3 = new g2o::VertexSE3Expmap();
            v_se3->setId(id);
            if (id == 0) {
                v_se3->setFixed(true);
            }
            v_se3->setEstimate(pose);
            optimizer_.addVertex(v_se3);
            cout << "pose added" << endl;
            return true;
        }
        */

        bool addPoseVertex(int id, double x, double y, double z, double q1, double q2, double q3, double q4){
            Vector3d trans(x, y, z);
            g2o::SE3Quat pose; //(R, t);
            pose.setTranslation(trans);
            g2o::VertexSE3Expmap* v_se3 = new g2o::VertexSE3Expmap();
            v_se3->setId(id);
            if (id == 0) {
                v_se3->setFixed(true);
            }
            v_se3->setEstimate(pose);
            optimizer_.addVertex(v_se3);
            cout << "pose added" << endl;
            return true;
        }


        bool addPointVertex(int id, double x, double y, double z){
            g2o::VertexPointXYZ* v_p = new g2o::VertexPointXYZ();
            v_p->setId(id);
            v_p->setMarginalized(true);
            v_p->setEstimate(Vector3d(x, y, z));
            optimizer_.addVertex(v_p);
            cout << "point added" << endl;
            return true;
        }

        bool addPosePointEdge(int pose_id, int point_id, double u, double v){

            //EdgeSE3ProjectXYZ
            g2o::EdgeProjectXYZ2UV* e = new g2o::EdgeProjectXYZ2UV();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                                optimizer_.vertices().find(pose_id)->second));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                                optimizer_.vertices().find(point_id)->second));
            e->setMeasurement(Vector2d(u, v));
            e->information() = Matrix2d::Identity();
            if (ROBUST_KERNEL) {
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
            }
            e->setParameterId(0, 0);
            optimizer_.addEdge(e);
            cout << "edge added" << endl;
            return true;
        }

        bool optimize(int itr){
            cout << "optimizing ..." << endl;
            optimizer_.save("tutorial_before_rand.txt");
            optimizer_.initializeOptimization();
            optimizer_.setVerbose(true);
            if (STRUCTURE_ONLY) {
                g2o::StructureOnlySolver<3> structure_only_ba;
                cout << "Performing structure-only BA:" << endl;
                g2o::OptimizableGraph::VertexContainer points;
                for (g2o::OptimizableGraph::VertexIDMap::const_iterator it =
                        optimizer_.vertices().begin();
                    it != optimizer_.vertices().end(); ++it) {
                g2o::OptimizableGraph::Vertex* v =
                    static_cast<g2o::OptimizableGraph::Vertex*>(it->second);
                if (v->dimension() == 3) points.push_back(v);
                }
                structure_only_ba.calc(points, itr);
            }
            cout << endl;
            cout << "Performing full BA:" << endl;
            optimizer_.optimize(itr);
            optimizer_.save("tutorial_after_rand.txt");
            cout << endl;
            return true;

        }


        //members
        g2o::SparseOptimizer optimizer_;
        g2o::CameraParameters* cam_params_;

};