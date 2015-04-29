#include <stdio.h>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>

#include "core/ChFileutils.h"
#include "core/ChStream.h"
#include "collision/ChCConvexDecomposition.h"
#include "collision/ChCCollisionModel.h"
#include "geometry/ChCTriangleMeshConnected.h"
#include "assets/ChTriangleMeshShape.h"
#include "assets/ChCylinderShape.h"
#include "assets/ChBoxShape.h"
#include "physics/ChSystem.h"

#include "config.h"
#include "subsys/ChVehicleModelData.h"
#include "subsys/vehicle/Vehicle.h"
#include "subsys/powertrain/SimplePowertrain.h"

using std::cout;
using std::endl;

// =============================================================================

// JSON file for vehicle model
std::string vehicle_file("hmmwv/vehicle/HMMWV_Vehicle_4WD.json");

// JSON files for powertrain (simple)
std::string simplepowertrain_file("hmmwv/powertrain/HMMWV_SimplePowertrain.json");

std::string lugged_file("hmmwv/lugged_wheel_section.obj");

// -----------------------------------------------------------------------------
// Simulation parameters
// -----------------------------------------------------------------------------

// Desired number of OpenMP threads (will be clamped to maximum available)
int threads = 2;

// Perform dynamic tuning of number of threads?
bool thread_tuning = true;

// Simulation duration.
double time_end = 16;

// Solver parameters
double time_step = 1e-3;

double tolerance = 1e-3;

int max_iteration_normal = 0;
int max_iteration_sliding = 40;
int max_iteration_spinning = 0;

// This should be faster than the vehicle when it hits the water
float contact_recovery_speed = 3;

// Output
const std::string out_dir = "../HMMWV";
const std::string pov_dir = out_dir + "/POVRAY";

int out_fps = 60;

// Continuous loop (only if OpenGL available)
bool loop = false;
chrono::ChSharedPtr<chrono::ChBody> chassisBody;

// fluid body radius
double fluid_r = 0.0175;

// Container dimensions
const double conversion = .3048;  // meters per foot
// note that when specifying dimensions for the geometry half lengths are used.
double dim_a = 16 * conversion * 0.5;  // length of end platforms default : 28.5
double dim_b = 5 * conversion * 0.5;     // length of slope at top
double dim_c = 12 * conversion * 0.5;    // length of submerged slope default: 48.5
double dim_d = 15 * conversion * 0.5;    // length of bottom default: 100
double dim_e = 8 * conversion * 0.5;     // full depth of trench
double dim_w = 14 * conversion * 0.5;    // width of trench default: 20
double dim_t = 5.0 / 12.0 * conversion;  // wall thickness default : 10

// Initial vehicle position and orientation
chrono::ChVector<> initLoc(-(dim_d + (dim_b + dim_c) * 2 + dim_a * .9), 0, dim_e * 2 + 0.65);
chrono::ChQuaternion<> initRot(1, 0, 0, 0);

// Variables that store convex meshes for reuse
chrono::collision::ChConvexDecompositionHACDv2 lugged_convex;
chrono::geometry::ChTriangleMeshConnected lugged_mesh;

// =============================================================================

class MyVehicle {
 public:
  MyVehicle(chrono::ChSystem* system);

  void Update(double time);

  chrono::ChSharedPtr<chrono::Vehicle> m_vehicle;
  chrono::ChSharedPtr<chrono::SimplePowertrain> m_powertrain;
  chrono::ChTireForces m_tire_forces;
};

void LoadConvexMesh(const std::string& file_name,
                    chrono::geometry::ChTriangleMeshConnected& convex_mesh,
                    chrono::collision::ChConvexDecompositionHACDv2& convex_shape,
                    const chrono::ChVector<>& pos = chrono::ChVector<>(0, 0, 0),
                    const chrono::ChQuaternion<>& rot = chrono::ChQuaternion<>(1, 0, 0, 0),
                    int hacd_maxhullcount = 1024,
                    int hacd_maxhullmerge = 256,
                    int hacd_maxhullvertexes = 64,
                    double hacd_concavity = 0.01,
                    double hacd_smallclusterthreshold = 0.0,
                    double hacd_fusetolerance = 1e-6) {
  convex_mesh.LoadWavefrontMesh(file_name, true, false);

  for (int i = 0; i < convex_mesh.m_vertices.size(); i++) {
    convex_mesh.m_vertices[i] = pos + rot.Rotate(convex_mesh.m_vertices[i]);
  }

  convex_shape.Reset();
  convex_shape.AddTriangleMesh(convex_mesh);
  convex_shape.SetParameters(hacd_maxhullcount, hacd_maxhullmerge, hacd_maxhullvertexes, hacd_concavity,
                             hacd_smallclusterthreshold, hacd_fusetolerance);
  convex_shape.ComputeConvexDecomposition();
}

void AddConvexCollisionModel(chrono::ChSharedPtr<chrono::ChBody> body,
                             chrono::geometry::ChTriangleMeshConnected& convex_mesh,
                             chrono::collision::ChConvexDecompositionHACDv2& convex_shape,
                             const chrono::ChVector<>& pos = chrono::ChVector<>(0, 0, 0),
                             const chrono::ChQuaternion<>& rot = chrono::ChQuaternion<>(1, 0, 0, 0),
                             bool use_original_asset = true) {
  chrono::collision::ChConvexDecomposition* used_decomposition = &convex_shape;

  int hull_count = used_decomposition->GetHullCount();

  for (int c = 0; c < hull_count; c++) {
    std::vector<chrono::ChVector<double> > convexhull;
    used_decomposition->GetConvexHullResult(c, convexhull);

    ((chrono::collision::ChCollisionModel*)body->GetCollisionModel())->AddConvexHull(convexhull, pos, rot);
    // Add each convex chunk as a new asset
    if (!use_original_asset) {
      std::stringstream ss;
      ss << convex_mesh.GetFileName() << "_" << c;
      chrono::geometry::ChTriangleMeshConnected trimesh_convex;
      used_decomposition->GetConvexHullResult(c, trimesh_convex);

      chrono::ChSharedPtr<chrono::ChTriangleMeshShape> trimesh_shape(new chrono::ChTriangleMeshShape);
      trimesh_shape->SetMesh(trimesh_convex);
      trimesh_shape->SetName(ss.str());
      trimesh_shape->Pos = pos;
      trimesh_shape->Rot = rot;
      body->GetAssets().push_back(trimesh_shape);
    }
  }
  // Add the original triangle mesh as asset
  if (use_original_asset) {
    chrono::ChSharedPtr<chrono::ChTriangleMeshShape> trimesh_shape(new chrono::ChTriangleMeshShape);
    trimesh_shape->SetMesh(convex_mesh);
    trimesh_shape->SetName(convex_mesh.GetFileName());
    trimesh_shape->Pos = chrono::VNULL;
    trimesh_shape->Rot = chrono::QUNIT;
    body->GetAssets().push_back(trimesh_shape);
  }
}

void AddCylinderGeometry(chrono::ChBody* body,
                         double radius,
                         double hlen,
                         const chrono::ChVector<>& pos = chrono::ChVector<>(0, 0, 0),
                         const chrono::ChQuaternion<>& rot = chrono::ChQuaternion<>(1, 0, 0, 0),
                         bool visualization = true) {
  body->GetCollisionModel()->AddCylinder(radius, radius, hlen, pos, rot);

  if (visualization) {
    chrono::ChSharedPtr<chrono::ChCylinderShape> cylinder(new chrono::ChCylinderShape);
    cylinder->GetCylinderGeometry().rad = radius;
    cylinder->GetCylinderGeometry().p1 = chrono::ChVector<>(0, hlen, 0);
    cylinder->GetCylinderGeometry().p2 = chrono::ChVector<>(0, -hlen, 0);
    cylinder->Pos = pos;
    cylinder->Rot = rot;
    body->GetAssets().push_back(cylinder);
  }
}

void AddBoxGeometry(chrono::ChBody* body,
                    const chrono::ChVector<>& size,
                    const chrono::ChVector<>& pos = chrono::ChVector<>(0, 0, 0),
                    const chrono::ChQuaternion<>& rot = chrono::ChQuaternion<>(1, 0, 0, 0),
                    bool visualization = true) {
  body->GetCollisionModel()->AddBox(size.x, size.y, size.z, pos, rot);

  if (visualization) {
    chrono::ChSharedPtr<chrono::ChBoxShape> box(new chrono::ChBoxShape);
    box->GetBoxGeometry().Size = size;
    box->Pos = pos;
    box->Rot = rot;
    body->GetAssets().push_back(box);
  }
}

void InitializeObject(chrono::ChSharedPtr<chrono::ChBody> body,
                      double mass,
                      chrono::ChSharedPtr<chrono::ChMaterialSurfaceBase> mat,
                      const chrono::ChVector<>& pos = chrono::ChVector<>(0, 0, 0),
                      const chrono::ChQuaternion<>& rot = chrono::ChQuaternion<>(1, 0, 0, 0),
                      bool collide = true,
                      bool fixed = false,
                      int collision_family = 2,
                      int do_not_collide_with = 4) {
  body->SetMass(mass);
  body->SetPos(pos);
  body->SetRot(rot);
  body->SetCollide(collide);
  body->SetBodyFixed(fixed);
  body->SetMaterialSurface(mat);
  body->GetCollisionModel()->ClearModel();
  body->GetCollisionModel()->SetFamily(collision_family);
  body->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(do_not_collide_with);
}

void FinalizeObject(chrono::ChSharedPtr<chrono::ChBody> body, chrono::ChSystem* system) {
  // Infer system type and contact method.
  chrono::ChBody::ContactMethod contact_method = body->GetContactMethod();
  body->GetCollisionModel()->BuildModel();
  system->AddBody(body);
}

MyVehicle::MyVehicle(chrono::ChSystem* system) {
  // Create and initialize the vehicle system
  LoadConvexMesh(chrono::vehicle::GetDataFile(lugged_file), lugged_mesh, lugged_convex);
  printf("Wheel Hulls: %d\n", lugged_convex.GetHullCount());

  m_vehicle =
      chrono::ChSharedPtr<chrono::Vehicle>(new chrono::Vehicle(system, chrono::vehicle::GetDataFile(vehicle_file)));
  m_vehicle->Initialize(chrono::ChCoordsys<>(initLoc, initRot));

  // Create and initialize the powertrain system
  m_powertrain = chrono::ChSharedPtr<chrono::SimplePowertrain>(
      new chrono::SimplePowertrain(chrono::vehicle::GetDataFile(simplepowertrain_file)));
  m_powertrain->Initialize();

  // Add contact geometry to the vehicle wheel bodies
  int numAxles = m_vehicle->GetNumberAxles();
  int numWheels = 2 * numAxles;

  for (int i = 0; i < numWheels; i++) {
    double radius = m_vehicle->GetWheel(i)->GetRadius();
    double width = m_vehicle->GetWheel(i)->GetWidth();

    chrono::ChSharedPtr<chrono::ChBody> wheelBody = m_vehicle->GetWheelBody(i);
    wheelBody->GetAssets().clear();

    wheelBody->GetCollisionModel()->ClearModel();
    for (int j = 0; j < 15; j++) {
      AddConvexCollisionModel(wheelBody, lugged_mesh, lugged_convex, chrono::VNULL,
                              chrono::Q_from_AngAxis(j * 24 * chrono::CH_C_DEG_TO_RAD, chrono::VECT_Y), false);
    }
    // This cylinder acts like the rims
    AddCylinderGeometry(wheelBody.get_ptr(), 0.223, 0.126);

    wheelBody->GetCollisionModel()->BuildModel();
    wheelBody->GetCollisionModel()->SetFamily(4);
    wheelBody->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(4);

    wheelBody->SetCollide(true);
    wheelBody->GetMaterialSurface()->SetFriction(0.8f);
  }

  // The vector of tire forces is required by the ChronoVehicle API. Since we
  // use rigid contact for tire-terrain interaction, these are always zero.
  m_tire_forces.resize(numWheels);

  chrono::ChSharedPtr<chrono::ChBody> chassisBody = m_vehicle->GetChassis();
  for (int i = 0; i < chassisBody->GetAssets().size(); i++) {
    ((chrono::ChVisualization*)chassisBody->GetAssets().at(i).get_ptr())->Pos = chrono::ChVector<>(0);
  }
  chassisBody->SetCollide(false);
}

void MyVehicle::Update(double time) {
  // Calculate driver inputs at current time
  double throttle = 0;
  double steering = 0;
  double braking = 0;

  // if (time > 0.5)
  //   throttle = 1.0 * .25;
  // else if (time > 0.25)
  //   throttle = 4 * (time - 0.25) * .25;

  if (time > 0.1) {
    throttle = 1;
  }
  std::cout << "Vehicle Speed:   " << m_vehicle->GetVehicleSpeed() << std::endl;

  // Update the powertrain system
  m_powertrain->Update(time, throttle, m_vehicle->GetDriveshaftSpeed());

  // Update the vehicle system.
  m_vehicle->Update(time, steering, braking, m_powertrain->GetOutputTorque(), m_tire_forces);
}

void SetupSystem(chrono::ChSystem* system) {
  system->Set_G_acc(chrono::ChVector<>(0, 0, -9.81));
  system->SetIntegrationType(chrono::ChSystem::INT_ANITESCU);
  system->SetLcpSolverType(chrono::ChSystem::LCP_ITERATIVE_SOR);
  system->SetIterLCPmaxItersSpeed(150);
  system->SetIterLCPmaxItersStab(150);
  system->SetMaxPenetrationRecoverySpeed(4.0);
}

void CreateContainer(chrono::ChSystem* system) {
  double dim_slope = sqrt(dim_e * dim_e + (dim_b + dim_c) * (dim_b + dim_c));
  double angle = atan(dim_e / (dim_b + dim_c));
  double width = dim_w + dim_t * 2.0;

  double container_friction = 1;

  chrono::ChSharedBodyPtr bottom_plate = chrono::ChSharedBodyPtr(new chrono::ChBody());
  chrono::ChSharedBodyPtr side_plate_1 = chrono::ChSharedBodyPtr(new chrono::ChBody());
  chrono::ChSharedBodyPtr side_plate_2 = chrono::ChSharedBodyPtr(new chrono::ChBody());
  chrono::ChSharedBodyPtr end_plate_1 = chrono::ChSharedBodyPtr(new chrono::ChBody());
  chrono::ChSharedBodyPtr end_plate_2 = chrono::ChSharedBodyPtr(new chrono::ChBody());
  chrono::ChSharedBodyPtr end_slope_1 = chrono::ChSharedBodyPtr(new chrono::ChBody());
  chrono::ChSharedBodyPtr end_slope_2 = chrono::ChSharedBodyPtr(new chrono::ChBody());

  chrono::ChSharedPtr<chrono::ChMaterialSurface> material =
      chrono::ChSharedPtr<chrono::ChMaterialSurface>(new chrono::ChMaterialSurface);
  material->SetFriction(container_friction);

  chrono::ChVector<> c_pos = chrono::VNULL;
  InitializeObject(bottom_plate, 1, material, chrono::VNULL + c_pos, chrono::QUNIT, true, true, 2, 2);
  InitializeObject(side_plate_1, 1, material, chrono::VNULL + c_pos, chrono::QUNIT, true, true, 2, 2);
  InitializeObject(side_plate_2, 1, material, chrono::VNULL + c_pos, chrono::QUNIT, true, true, 2, 2);
  InitializeObject(end_plate_1, 1, material, chrono::VNULL + c_pos, chrono::QUNIT, true, true, 2, 2);
  InitializeObject(end_plate_2, 1, material, chrono::VNULL + c_pos, chrono::QUNIT, true, true, 2, 2);
  InitializeObject(end_slope_1, 1, material, chrono::VNULL + c_pos, chrono::QUNIT, true, true, 2, 2);
  InitializeObject(end_slope_2, 1, material, chrono::VNULL + c_pos, chrono::QUNIT, true, true, 2, 2);

  // Bottom plate
  AddBoxGeometry(bottom_plate.get_ptr(), chrono::ChVector<>(dim_d * 1.1, width * 1.1, dim_t), chrono::VNULL,
                 chrono::QUNIT);
  // Side walls
  AddBoxGeometry(side_plate_1.get_ptr(),
                 chrono::ChVector<>(dim_d + (dim_b + dim_c + dim_a) * 2, dim_t, dim_e * 2 + dim_t * 2),
                 chrono::ChVector<>(0, +(dim_w + dim_t), dim_e * 2), chrono::QUNIT);
  AddBoxGeometry(side_plate_2.get_ptr(),
                 chrono::ChVector<>(dim_d + (dim_b + dim_c + dim_a) * 2, dim_t, dim_e * 2 + dim_t * 2),
                 chrono::ChVector<>(0, -(dim_w + dim_t), dim_e * 2), chrono::QUNIT);
  // End Platforms
  AddBoxGeometry(end_plate_1.get_ptr(), chrono::ChVector<>(dim_a, width, dim_t),
                 chrono::ChVector<>(+(dim_d + dim_c * 2 + dim_b * 2 + dim_a), 0, dim_e * 2.0), chrono::QUNIT);
  AddBoxGeometry(end_plate_2.get_ptr(), chrono::ChVector<>(dim_a, width, dim_t),
                 chrono::ChVector<>(-(dim_d + dim_c * 2 + dim_b * 2 + dim_a), 0, dim_e * 2.0), chrono::QUNIT);
  // Slopes
  AddBoxGeometry(end_slope_1.get_ptr(), chrono::ChVector<>(dim_slope, dim_w + dim_t * 2.0, dim_t),
                 chrono::ChVector<>(+(dim_d + (dim_c + dim_b) + sin(angle) * dim_t * 0.5), 0, dim_e),
                 chrono::Q_from_AngAxis(-angle, chrono::VECT_Y));
  AddBoxGeometry(end_slope_2.get_ptr(), chrono::ChVector<>(dim_slope, dim_w + dim_t * 2.0, dim_t),
                 chrono::ChVector<>(-(dim_d + (dim_c + dim_b) + sin(angle) * dim_t * 0.5), 0, dim_e),
                 chrono::Q_from_AngAxis(+angle, chrono::VECT_Y));

  // Top
  AddBoxGeometry(bottom_plate.get_ptr(), chrono::ChVector<>(dim_d + (dim_b + dim_c + dim_a) * 2, width, dim_t),
                 chrono::ChVector<>(0, 0, dim_e * 4 + dim_t * 2.0), chrono::QUNIT);

  // End Caps
  AddBoxGeometry(end_plate_1.get_ptr(), chrono::ChVector<>(dim_t, width, dim_e + dim_t * 2.0),
                 chrono::ChVector<>(+(dim_d + dim_c * 2 + dim_b * 2 + dim_a * 2), 0, dim_e * 3.0 + dim_t),
                 chrono::QUNIT);
  AddBoxGeometry(end_plate_2.get_ptr(), chrono::ChVector<>(dim_t, width, dim_e + dim_t * 2.0),
                 chrono::ChVector<>(-(dim_d + dim_c * 2 + dim_b * 2 + dim_a * 2), 0, dim_e * 3.0 + dim_t),
                 chrono::QUNIT);

  FinalizeObject(bottom_plate, system);
  FinalizeObject(side_plate_1, system);
  FinalizeObject(side_plate_2, system);
  FinalizeObject(end_plate_1, system);
  FinalizeObject(end_plate_2, system);
  FinalizeObject(end_slope_1, system);
  FinalizeObject(end_slope_2, system);
}
// =============================================================================
int main(int argc, char* argv[]) {
  // Set path to ChronoVehicle data files
  chrono::vehicle::SetDataPath(CHRONOVEHICLE_DATA_DIR);

  // Create system.
  chrono::ChSystem* system = new chrono::ChSystem();
  SetupSystem(system);
  // Create the container for vehicle
  CreateContainer(system);
  // Create and initialize the vehicle systems
  MyVehicle vehicle(system);

  // Run simulation for specified time.
  int out_steps = std::ceil((1.0 / time_step) / out_fps);

  double time = 0;
  int sim_frame = 0;
  int out_frame = 0;
  int next_out_frame = 0;
  double exec_time = 0;
  int num_contacts = 0;

  while (time < time_end) {
    // vehicle.m_vehicle->GetChassis()->Empty_forces_accumulators();
    // [force, point applied, false=in global frame]
    // vehicle.m_vehicle->GetChassis()->Accumulate_force(chrono::ChVector<>(frc.x,frc.y,frc.z),
    // chrono::ChVector<>(cpt.x,cpt.y,cpt.z),
    // false);
    // chrono::ChVector<> position = vehicle.m_vehicle->GetChassis()->GetPos();
    // chrono::ChVector velocity =   vehicle.m_vehicle->GetChassis()->GetPos_dt();
    // chrono::ChQuaternion<> rotation = vehicle.m_vehicle->GetChassis()->GetRot();
    // chrono::ChVector<> omega = vehicle.m_vehicle->GetChassis()->GetWvel_loc();
    // For the wheels do this: index 0, 1, 2,3 for the 4 wheels,
    //  vehicle.m_vehicle->GetWheelBody(0)->GetPos();

    system->DoStepDynamics(time_step);

    vehicle.Update(time);
    // Update counters.

    std::cout << "time: " << time << " N Contacts" << system->GetNcontacts() << std::endl;

    time += time_step;
    sim_frame++;
    exec_time += system->GetTimerStep();
  }
  return 0;
}
