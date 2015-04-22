#include <stdio.h>
#include <vector>
#include <cmath>
#include <string>

#include "core/ChFileutils.h"
#include "core/ChStream.h"
#include "collision/ChCConvexDecomposition.h"

#include "chrono_parallel/physics/ChSystemParallel.h"
#include "chrono_parallel/lcp/ChLcpSystemDescriptorParallel.h"
#include "chrono_parallel/collision/ChCNarrowphaseRUtils.h"

#include "chrono_utils/ChUtilsGeometry.h"
#include "chrono_utils/ChUtilsCreators.h"
#include "chrono_utils/ChUtilsInputOutput.h"
#include "chrono_utils/ChUtilsGenerators.h"
#include "chrono_utils/ChUtilsVehicle.h"

#include "config.h"
#include "subsys/ChVehicleModelData.h"
#include "subsys/vehicle/Vehicle.h"
#include "subsys/powertrain/SimplePowertrain.h"

//#include "input_output.h"

//#undef CHRONO_PARALLEL_HAS_OPENGL

//#define FILLING
#ifdef CHRONO_PARALLEL_HAS_OPENGL
#include "chrono_opengl/ChOpenGLWindow.h"
#endif

using namespace chrono;
using namespace chrono::collision;
using namespace chrono::geometry;
using namespace chrono::utils;
using std::cout;
using std::endl;

// =============================================================================

// JSON file for vehicle model
std::string vehicle_file("hmmwv/vehicle/HMMWV_Vehicle.json");

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
ChSharedPtr<ChBody> chassisBody;

// fluid body radius
double fluid_r = 0.0175;

// Container dimensions
const real conversion = .3048;  // meters per foot
// note that when specifying dimensions for the geometry half lengths are used.
real dim_a = 28.5 * conversion * 0.5;  // length of end platforms
real dim_b = 5 * conversion * 0.5;     // length of slope at top
real dim_c = 12 * conversion * 0.5;    // length of submerged slope default: 48.5
real dim_d = 15 * conversion * 0.5;    // length of bottom default: 100
real dim_e = 8 * conversion * 0.5;     // full depth of trench
real dim_w = 14 * conversion * 0.5;    // width of trench default: 20
real dim_t = 5.0 / 12.0 * conversion;  // wall thickness default : 10

// Initial vehicle position and orientation
ChVector<> initLoc(-(dim_d + (dim_b + dim_c) * 2 + dim_a), 0, dim_e * 2 + 0.8);
ChQuaternion<> initRot(1, 0, 0, 0);

// Variables that store convex meshes for reuse
ChConvexDecompositionHACDv2 lugged_convex;
ChTriangleMeshConnected lugged_mesh;

// =============================================================================

class MyVehicle {
 public:
  MyVehicle(ChSystem* system);

  void Update(double time);

  ChSharedPtr<Vehicle> m_vehicle;
  ChSharedPtr<SimplePowertrain> m_powertrain;
  ChTireForces m_tire_forces;
};

MyVehicle::MyVehicle(ChSystem* system) {
  // Create and initialize the vehicle system
  LoadConvexMesh(vehicle::GetDataFile(lugged_file), lugged_mesh, lugged_convex);
  printf("Wheel Hulls: %d\n", lugged_convex.GetHullCount());

  m_vehicle = ChSharedPtr<Vehicle>(new Vehicle(system, vehicle::GetDataFile(vehicle_file)));
  m_vehicle->Initialize(ChCoordsys<>(initLoc, initRot));

  // Create and initialize the powertrain system
  m_powertrain = ChSharedPtr<SimplePowertrain>(new SimplePowertrain(vehicle::GetDataFile(simplepowertrain_file)));
  m_powertrain->Initialize();

  // Add contact geometry to the vehicle wheel bodies
  int numAxles = m_vehicle->GetNumberAxles();
  int numWheels = 2 * numAxles;

  for (int i = 0; i < numWheels; i++) {
    double radius = m_vehicle->GetWheel(i)->GetRadius();
    double width = m_vehicle->GetWheel(i)->GetWidth();

    ChSharedPtr<ChBody> wheelBody = m_vehicle->GetWheelBody(i);
    wheelBody->GetAssets().clear();

    wheelBody->GetCollisionModel()->ClearModel();
    for (int j = 0; j < 15; j++) {
      AddConvexCollisionModel(wheelBody, lugged_mesh, lugged_convex, VNULL,
                              Q_from_AngAxis(j * 24 * CH_C_DEG_TO_RAD, VECT_Y), false);
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

  ChSharedPtr<ChBody> chassisBody = m_vehicle->GetChassis();
  for (int i = 0; i < chassisBody->GetAssets().size(); i++) {
    ((ChVisualization*)chassisBody->GetAssets().at(i).get_ptr())->Pos = ChVector<>(0);
  }
  chassisBody->SetCollide(false);
}

void MyVehicle::Update(double time) {
  // Calculate driver inputs at current time
  double throttle = 0;
  double steering = 0;
  double braking = 0;

  if (time > 0.5)
    throttle = 1.0 * .25;
  else if (time > 0.25)
    throttle = 4 * (time - 0.25) * .25;

  // Update the powertrain system
  m_powertrain->Update(time, throttle, m_vehicle->GetDriveshaftSpeed());

  // Update the vehicle system.
  m_vehicle->Update(time, steering, braking, m_powertrain->GetOutputTorque(), m_tire_forces);
}

void SetupSystem(ChSystem* system) {
  system->Set_G_acc(ChVector<>(0, 0, -9.81));
  omp_set_num_threads(2);
  system->SetIntegrationType(ChSystem::INT_ANITESCU);
  system->SetLcpSolverType(ChSystem::LCP_ITERATIVE_SOR);
  system->SetIterLCPmaxItersSpeed(150);
  system->SetIterLCPmaxItersStab(150);
  system->SetMaxPenetrationRecoverySpeed(4.0);
}

void CreateContainer(ChSystem* system) {
  real dim_slope = sqrt(dim_e * dim_e + (dim_b + dim_c) * (dim_b + dim_c));
  real angle = atan(dim_e / (dim_b + dim_c));
  real width = dim_w + dim_t * 2.0;

  real container_friction = 1;

  ChSharedBodyPtr bottom_plate = ChSharedBodyPtr(new ChBody());
  ChSharedBodyPtr side_plate_1 = ChSharedBodyPtr(new ChBody());
  ChSharedBodyPtr side_plate_2 = ChSharedBodyPtr(new ChBody());
  ChSharedBodyPtr end_plate_1 = ChSharedBodyPtr(new ChBody());
  ChSharedBodyPtr end_plate_2 = ChSharedBodyPtr(new ChBody());
  ChSharedBodyPtr end_slope_1 = ChSharedBodyPtr(new ChBody());
  ChSharedBodyPtr end_slope_2 = ChSharedBodyPtr(new ChBody());

  ChSharedPtr<ChMaterialSurface> material = ChSharedPtr<ChMaterialSurface>(new ChMaterialSurface);
  material->SetFriction(container_friction);

  Vector c_pos = VNULL;
  InitializeObject(bottom_plate, 1, material, VNULL + c_pos, QUNIT, true, true, 2, 2);
  InitializeObject(side_plate_1, 1, material, VNULL + c_pos, QUNIT, true, true, 2, 2);
  InitializeObject(side_plate_2, 1, material, VNULL + c_pos, QUNIT, true, true, 2, 2);
  InitializeObject(end_plate_1, 1, material, VNULL + c_pos, QUNIT, true, true, 2, 2);
  InitializeObject(end_plate_2, 1, material, VNULL + c_pos, QUNIT, true, true, 2, 2);
  InitializeObject(end_slope_1, 1, material, VNULL + c_pos, QUNIT, true, true, 2, 2);
  InitializeObject(end_slope_2, 1, material, VNULL + c_pos, QUNIT, true, true, 2, 2);

  // Bottom plate
  AddBoxGeometry(bottom_plate.get_ptr(), Vector(dim_d * 1.1, width * 1.1, dim_t), VNULL, QUNIT);
  // Side walls
  AddBoxGeometry(side_plate_1.get_ptr(), Vector(dim_d + (dim_b + dim_c + dim_a) * 2, dim_t, dim_e * 2 + dim_t * 2),
                 Vector(0, +(dim_w + dim_t), dim_e * 2), QUNIT);
  AddBoxGeometry(side_plate_2.get_ptr(), Vector(dim_d + (dim_b + dim_c + dim_a) * 2, dim_t, dim_e * 2 + dim_t * 2),
                 Vector(0, -(dim_w + dim_t), dim_e * 2), QUNIT);
  // End Platforms
  AddBoxGeometry(end_plate_1.get_ptr(), Vector(dim_a, width, dim_t),
                 Vector(+(dim_d + dim_c * 2 + dim_b * 2 + dim_a), 0, dim_e * 2.0), QUNIT);
  AddBoxGeometry(end_plate_2.get_ptr(), Vector(dim_a, width, dim_t),
                 Vector(-(dim_d + dim_c * 2 + dim_b * 2 + dim_a), 0, dim_e * 2.0), QUNIT);
  // Slopes
  AddBoxGeometry(end_slope_1.get_ptr(), Vector(dim_slope, dim_w + dim_t * 2.0, dim_t),
                 Vector(+(dim_d + (dim_c + dim_b) + sin(angle) * dim_t * 0.5), 0, dim_e),
                 Q_from_AngAxis(-angle, VECT_Y));
  AddBoxGeometry(end_slope_2.get_ptr(), Vector(dim_slope, dim_w + dim_t * 2.0, dim_t),
                 Vector(-(dim_d + (dim_c + dim_b) + sin(angle) * dim_t * 0.5), 0, dim_e),
                 Q_from_AngAxis(+angle, VECT_Y));

  // Top
  AddBoxGeometry(bottom_plate.get_ptr(), Vector(dim_d + (dim_b + dim_c + dim_a) * 2, width, dim_t),
                 Vector(0, 0, dim_e * 4 + dim_t * 2.0), QUNIT);

  // End Caps
  AddBoxGeometry(end_plate_1.get_ptr(), Vector(dim_t, width, dim_e + dim_t * 2.0),
                 Vector(+(dim_d + dim_c * 2 + dim_b * 2 + dim_a * 2), 0, dim_e * 3.0 + dim_t), QUNIT);
  AddBoxGeometry(end_plate_2.get_ptr(), Vector(dim_t, width, dim_e + dim_t * 2.0),
                 Vector(-(dim_d + dim_c * 2 + dim_b * 2 + dim_a * 2), 0, dim_e * 3.0 + dim_t), QUNIT);

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
  vehicle::SetDataPath(CHRONOVEHICLE_DATA_DIR);

  // Create system.
  ChSystem* system = new ChSystem();
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
#ifdef CHRONO_PARALLEL_HAS_OPENGL
  // Initialize OpenGL
  opengl::ChOpenGLWindow& gl_window = opengl::ChOpenGLWindow::getInstance();
  gl_window.Initialize(1280, 720, "Fording Simulation", system);
  gl_window.SetCamera(ChVector<>(0, -10, 0), ChVector<>(0, 0, 0), ChVector<>(0, 0, 1), 0.1f);
  gl_window.Pause();

  while (gl_window.Active()) {
    if (gl_window.DoStepDynamics(time_step)) {
      vehicle.Update(time);
      time += time_step;
    }
    gl_window.Render();
  }
  exit(0);

#endif

  //  while (time < time_end) {
  //    TimingOutput(system);
  //    system->DoStepDynamics(time_step);
  //    vehicle.Update(time);
  //    // Update counters.
  //    time += time_step;
  //    sim_frame++;
  //    exec_time += system->GetTimerStep();
  //  }
  return 0;
}
