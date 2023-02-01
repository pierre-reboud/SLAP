// g2o - General Graph Optimization
// Copyright (C) 2011 R. Kuemmerle, G. Grisetti, W. Burgard
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <cmath>
#include <iostream>

#include "../libs/g2o/g2o/core/block_solver.h"
#include "../libs/g2o/g2o/core/factory.h"
#include "../libs/g2o/g2o/core/optimization_algorithm_factory.h"
#include "../libs/g2o/g2o/core/optimization_algorithm_gauss_newton.h"
#include "../libs/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "../libs/g2o/g2o/core/sparse_optimizer.h"
#include "../libs/g2o/g2o/solvers/eigen/linear_solver_eigen.h"


#include "../libs/g2o/g2o/stuff/sampler.h"

#include "../libs/g2o/g2o/types/slam3d/types_slam3d.h"

#include "../libs/g2o/g2o/types/slam3d_addons/types_slam3d_addons.h"

#include "../libs/g2o/g2o/core/eigen_types.h"

#include "../libs/g2o/g2o/types/slam2d/types_slam2d.h"

#include "../libs/g2o/g2o/types/sba/types_sba.h"
#include "../libs/g2o/g2o/types/sba/types_six_dof_expmap.h"

#include <stdint.h>
#include <unordered_set>

//#include "g2o/visual_slam3d/visual_slam3d.h" 
#include "visual_slam3d.h" 

#define OUTLIER_RATIO 0.0

#define PIXEL_NOISE 0.0



using namespace std;
using namespace g2o;

class Sample {
 public:
  static int uniform(int from, int to) {
    return static_cast<int>(g2o::Sampler::uniformRand(from, to));
  }
};

int main() {
  // visual SLAM
  cout << "---VISUAL SLAMMM--- " << endl;



  std::unique_ptr<VisualSLAM3d> visual_slam3d = std::make_unique<VisualSLAM3d>();

  visual_slam3d->setCameraParam(100, 100, 100);

  /////////////////////////////////////

  visual_slam3d->addPoseVertex(0, 0, 0, 0, 0, 0, 0, 1);
  visual_slam3d->addPoseVertex(1, 0, 0, 0, 0, 0, 0, 1);
  visual_slam3d->addPoseVertex(2, 0, 0, 0, 0, 0, 0, 1);


  visual_slam3d->addPointVertex(3, 0, 0, 6);
  visual_slam3d->addPointVertex(4, 0, 0, 6);

  Vector2d z03 = visual_slam3d->project(-2, 0, 6);
  Vector2d z13 = visual_slam3d->project(-2, 0, 4);
  Vector2d z23 = visual_slam3d->project(-2, 0, 2);

  Vector2d z04 = visual_slam3d->project(2, 0, 6);
  Vector2d z14 = visual_slam3d->project(2, 0, 4);
  Vector2d z24 = visual_slam3d->project(2, 0, 2);

  visual_slam3d->addPosePointEdge(0, 3, z03[0], z03[1]); //67, 100);
  visual_slam3d->addPosePointEdge(1, 3, z13[0], z13[1]); //50, 100);
  visual_slam3d->addPosePointEdge(2, 3, z23[0], z23[1]); // 1, 100);

  visual_slam3d->addPosePointEdge(0, 4, z04[0], z04[1]); // 133, 100);
  visual_slam3d->addPosePointEdge(1, 4, z14[0], z14[1]); // 150, 100);
  visual_slam3d->addPosePointEdge(2, 4, z24[0], z24[1]); // 199, 100);

  visual_slam3d->optimize(10);


/*
  EdgeSE3Expmap* odom = new EdgeSE3Expmap;
  odom->vertices()[0] = visual_slam3d->optimizer_.vertex(0);
  odom->vertices()[1] = visual_slam3d->optimizer_.vertex(1);
  Eigen::Quaterniond quaternion5(0, 0, 0, 1);
  Eigen::Vector3d position5(0, 0, 2);
  g2o::SE3Quat pose5;
  pose5.setTranslation(position5);
  odom->setMeasurement(pose5);
  visual_slam3d->optimizer_.addEdge(odom);


  EdgeSE3Expmap* odom2 = new EdgeSE3Expmap;
  odom2->vertices()[0] = visual_slam3d->optimizer_.vertex(1);
  odom2->vertices()[1] = visual_slam3d->optimizer_.vertex(2);
  Eigen::Quaterniond quaternion6(0, 0, 0, 1);
  Eigen::Vector3d position6(0, 0, 2);
  g2o::SE3Quat pose6;
  pose6.setTranslation(position6);
  odom2->setMeasurement(pose6);
  visual_slam3d->optimizer_.addEdge(odom2);
*/
  /*

  int vertex_id = 0;
  for (size_t i = 0; i < 15; ++i) {
    visual_slam3d->addPoseVertex(vertex_id, i * 0.04 - 1.0, 0, 0, 0, 0, 0, 1);

    Vector3d trans(i * 0.04 - 1., 0, 0);
    Eigen::Quaterniond q;
    q.setIdentity();
    g2o::SE3Quat pose(q, trans);

    true_poses.push_back(pose);
    vertex_id++;
  }
  int point_id = vertex_id;
  int point_num = 0;
  double sum_diff2 = 0;

  cout << endl;
  unordered_map<int, int> pointid_2_trueid;
  unordered_set<int> inliers;

  for (size_t i = 0; i < true_points.size(); ++i) {

    int num_obs = 0;
    for (size_t j = 0; j < true_poses.size(); ++j) {
      Vector2d z = cam_params->cam_map(true_poses.at(j).map(true_points.at(i)));
      if (z[0] >= 0 && z[1] >= 0 && z[0] < 640 && z[1] < 480) {
        ++num_obs;
      }
    }
    if (num_obs >= 2) {
      visual_slam3d->addPointVertex(point_id, true_points.at(i)[0] + g2o::Sampler::gaussRand(0., 1), 
                                              true_points.at(i)[1] + g2o::Sampler::gaussRand(0., 1), 
                                              true_points.at(i)[2] + g2o::Sampler::gaussRand(0., 1));
      bool inlier = true;
      for (size_t j = 0; j < true_poses.size(); ++j) {
        Vector2d z =
            cam_params->cam_map(true_poses.at(j).map(true_points.at(i)));

            cout << endl << "pose: " << true_poses.at(j) << " point: " << true_points.at(i) << " z: " << z << endl;

        if (z[0] >= 0 && z[1] >= 0 && z[0] < 640 && z[1] < 480) {
          double sam = g2o::Sampler::uniformRand(0., 1.);
          if (sam < OUTLIER_RATIO) {
            z = Vector2d(Sample::uniform(0, 640), Sample::uniform(0, 480));
            inlier = false;
          }
          z += Vector2d(g2o::Sampler::gaussRand(0., PIXEL_NOISE),
                        g2o::Sampler::gaussRand(0., PIXEL_NOISE));
          visual_slam3d->addPosePointEdge(j, point_id, z[0], z[1]);
        }
      }

      pointid_2_trueid.insert(make_pair(point_id, i));
      ++point_id;
      ++point_num;
    }
  }

*/

 

  

  return 0;
}
