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
/*
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
*/
#include <stdint.h>
#include <unordered_set>


#include "visual_slam3d.h" 

#define OUTLIER_RATIO 0.0

#define PIXEL_NOISE 0.0



using namespace std;
using namespace g2o;


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

 

  return 0;
}
