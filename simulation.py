import logging
from os import times
from typing import TYPE_CHECKING, Callable, cast

import numpy as np
from numpy.typing import NDArray


import porepy as pp
from porepy.applications.boundary_conditions.model_boundary_conditions import (
    BoundaryConditionsMechanicsNeumann,
    HydrostaticPressureValues,
)

from porepy.applications.initial_conditions.model_initial_conditions import (
    InitialConditionHydrostaticPressureValues,
)
from porepy.numerics.nonlinear import line_search

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# 2D box
class Geometry(pp.PorePyModel):

    def domain_sizes(self) -> NDArray[np.float64]:
        return self.units.convert_units(
            self.params.get("domain_sizes", np.ones(2, dtype=float)), "m"
        )

    def set_domain(self) -> None:
        x_size, y_size = self.domain_sizes()
        box = {
            "xmin": 0.0,
            "xmax": x_size,
            "ymin": -y_size,
            "ymax": 0.0,
        }
        self._domain = pp.Domain(box)
    
    # def set_fractures(self) -> None:
        """Setting a diagonal fracture"""
        dx, dy = self.domain_sizes()

        length = 2000.0
        height = 100.0

        x0 = 0.8 * dx
        y0 = -0.8 * dy  # <-- negative

        frac_1_points = self.units.convert_units(
            np.array([[x0, x0 + length],
                    [y0, y0 + height]]),
            "m",
        )

                
        frac_1 = pp.LineFracture(frac_1_points)

        
        self._fractures = [frac_1]

class BoundaryConditionsPressure(pp.PorePyModel):

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
     
        domain_sides = self.domain_boundary_sides(sd)

        # Defult neu on all sides. Set dir on east, noth and south.
        return pp.BoundaryCondition(sd, domain_sides.east + domain_sides.north + domain_sides.south, "dir")
    

    def bc_values_darcy_flux(self, bg: pp.BoundaryGrid) -> np.ndarray:

        mass_flux_vals = np.zeros(bg.num_cells)

        domain_sides = self.domain_boundary_sides(bg)

        # Boolean mask for cells on the west side of the domain, where we will set influx.
        influx_cells = domain_sides.west.copy()

    
        domain_depth = self.params.get("domain_sizes")[1]

        # Well in the middle of the domain. Set influx in a region around the well.
        well_depth = -domain_depth/2

        # Right now have to have big enough domain to make sure we have influx cells.
        # Update boolean mask to only include cells within a certain distance from the well depth.
        influx_cells &= bg.cell_centers[1] > well_depth - 1100
        influx_cells &= bg.cell_centers[1] < well_depth + 1100

   
        vals = self.params.get("injection_well_fluxes", 0.0)

        # Interpolate between values in schedule to get the current injection flux. 
        # The factor 1/2.63e6 is from converting m^3/month to m^3/s.
        # The negative sign is because the fluxes in the schedule are given as negative for inflow.
        tot_input_values = -1*1/2.63e6*self.get_well_value(vals, self.time_manager.schedule, self.time_manager.time)

        # Calculate total area of the influx cells. In 2D this are lentghs, in 3D it would be areas.
        # In 2D assume with of 1m.
        tot_volume = bg.cell_volumes[influx_cells].sum()

        # Make sure we dont divide by zero if there are no influx cells. Set input values to zero in this case.
        input_values_cell = 0.0 if tot_volume <= 0 else tot_input_values / tot_volume

        # Negative inflow values. Given by m^3 per month in the schedule, converted to m/s.
        mass_flux_vals[influx_cells] =  input_values_cell * bg.cell_volumes[influx_cells]

        return mass_flux_vals 

     # It returns the well value (pressure) corresponding to the current simulation time.
    def get_well_value(self, values, times, current_time):
        if current_time < times[0]:
            raise ValueError("Current time is before the start of the well protocol.")
        
        elif current_time > times[-1]:
            raise ValueError("Current time is after the end of the well protocol.")
        
        else:
            return float(np.interp(current_time, times, values))

    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
    
        domain_sides = self.domain_boundary_sides(bg)
        values = np.zeros(bg.num_cells)
        
        depth = self.depth(bg.cell_centers)

        values[domain_sides.east] = self.hydrostatic_pressure(depth[domain_sides.east])
        values[domain_sides.south] = self.hydrostatic_pressure(depth[domain_sides.south]) 
        values[domain_sides.north] = self.hydrostatic_pressure(depth[domain_sides.north])
    
        return values

class BoundaryConditionsMechanicsNeumann((pp.PorePyModel)):

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
       
        domain_sides = self.domain_boundary_sides(sd)
        bc = pp.BoundaryConditionVectorial(sd, domain_sides.all_bf, "neu")

        if sd.dim < self.nd:
            # No displacement is implemented on grids of co-dimension >= 1.
            return bc
    
        bc.internal_to_dirichlet(sd)

        # West side: Roller (ux = 0)
        # bc.is_dir[0, domain_sides.west] = True
        # bc.is_neu[0, domain_sides.west] = False

        # South side: Roller (uy = 0)
        #  bc.is_dir[1, domain_sides.north] = True
        # bc.is_neu[1, domain_sides.north] = False

       
        faces_to_fix = self.faces_to_fix(sd)

        dir = [
         np.array([False, True]), # Fix yon face 1 (west) in y. 
         np.array([False, True]),  # Fix y and x on face 2 (east). 
         np.array([True, False]),] # Fix y on face 3 (North).
        
        # ~(invert True/False). Set dir on one face.
        for i, face in enumerate(faces_to_fix):
            bc.is_dir[:, face] = dir[i]
            bc.is_neu[:, face] = ~dir[i]  
    
        return bc

    def faces_to_fix(self, sd: pp.Grid) -> list[np.int64]:

        domain_sides = self.domain_boundary_sides(sd)
        box = self.domain.bounding_box

        # Point 1 is on the center top of the west boundary, having min x coordinate and
        # y coordinate slightly smaller than the max y coordinate. This is intended to
        # avoid picking a face along the z-aligned edge.
        x_mean = 0.5 * (box["xmax"] + box["xmin"])
        # Compute a cell size h to place the point slightly inside the domain along the
        # y direction instead of at the very corner.

        h = np.mean(sd.face_areas[domain_sides.west])
        y_high = box["ymax"] - 0.5 * h


        point_1 = np.array([box["xmin"], y_high])
        pts = sd.face_centers[:2, domain_sides.west]
        ind_1 = domain_sides.west.nonzero()[0][
            np.argmin(pp.distances.point_pointset(point_1, pts))
        ]
        # Point 2 is on the center top of the east boundary.
        point_2 = np.array([box["xmax"], y_high])
        pts = sd.face_centers[:2, domain_sides.east]
        ind_2 = domain_sides.east.nonzero()[0][
            np.argmin(pp.distances.point_pointset(point_2, pts))
        ]
        # Point 3 is on the center top of the north boundary, having min y coordinate
        # and mean x coordinate.
        point_3 = np.array([x_mean, box["ymin"]])
        pts = sd.face_centers[:2, domain_sides.north]
        ind_3 = domain_sides.north.nonzero()[0][
            np.argmin(pp.distances.point_pointset(point_3, pts))
        ]

        return [ind_1, ind_2, ind_3]

class LithostaticBoundaryStressValues(pp.PorePyModel):

    @property
    def lithostatic_stress_multipliers(self) -> np.ndarray:

        return self.params.get("lithostatic_stress_multipliers", np.ones(2))

    # Values for naumann boundary conditions for stress.
    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
 
        # Initialize array for stress values.
        values = np.zeros((2, boundary_grid.num_cells))

        # Assume zero initial stress state.
        if self.time_manager.time < 1e-5:
            return values.ravel("F")

        gravity = self.gravity_force_magnitude("solid")

        # Multiply with lithostatic stress multipliers, which can be used to set
        # different stresses in different directions.
        gradient = self.lithostatic_stress_multipliers * gravity

        # Get domain sides and depth at boundary cell centers.
        domain_sides = self.domain_boundary_sides(boundary_grid)
        depth = self.depth(boundary_grid.cell_centers)

        # The sign of the stress depends on the side of the domain according to the
        # direction of the outer normal vector on the boundary. Loop over directions.
        # Positive in compression. 
        for i, sides in enumerate([["west", "east"], ["south", "north"]]):

            # Apply stress on both sides of the domain in direction i.
            for side, sign in zip(sides, [1, -1]):
                # Get indices of faces on the given side.
                ind = getattr(domain_sides, side)

                # Can add 3D and still work. 
                if np.any(ind):
                    # Set ith component of stress on these faces.
                    values[i, ind] = (
                        gradient[i]
                        * depth[ind]
                        * sign
                        * boundary_grid.cell_volumes[ind]
                    )

        return values.ravel("F")

class ExportFractureQuantities(pp.PorePyModel):
     
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._p_ref = None
        self._u_ref = None
  

    def data_to_export(self) -> list:
        data = super().data_to_export()

        INJECTION_TIME = 3 * pp.YEAR

        sds = self.mdg.subdomains(dim=2)
        p_off = np.cumsum([0] + [sd.num_cells for sd in sds])
        u_off = np.cumsum([0] + [sd.num_cells * self.nd for sd in sds])

        # Always evaluate once per call
        p = self.evaluate_and_scale(sds, "pressure", "Pa")
        u = self.evaluate_and_scale(sds, "displacement", "m")


        will_set = (self._p_ref is None and self.time_manager.time >= INJECTION_TIME)
        if will_set:
            self._p_ref = p.copy()
            self._u_ref = u.copy()

        if self.time_manager.time < INJECTION_TIME:
            dp = np.zeros_like(p)
            du = np.zeros_like(u)
        else:
            dp = p - self._p_ref
            du = u - self._u_ref

        for i, sd in enumerate(sds):
            data.append((sd, "delta_pressure_from_injection", dp[p_off[i]:p_off[i+1]]))
            data.append((sd, "delta_displacement_from_injection", du[u_off[i]:u_off[i+1]]))

        return data

class Simulation2D(
    # Domain
    Geometry,

    # Constitutive laws
    pp.constitutive_laws.GravityForce,
    pp.constitutive_laws.CubicLawPermeability,


    # Initial conditions
    InitialConditionHydrostaticPressureValues,

    # Boundary conditions stress
    LithostaticBoundaryStressValues,
    BoundaryConditionsMechanicsNeumann,
    
    # Boundary conditions pressure
    BoundaryConditionsPressure,
    HydrostaticPressureValues,

    ExportFractureQuantities,

    pp.models.solution_strategy.ContactIndicators,

    # Base class
    pp.Poromechanics,

):
    pass
 
# Add all parameters for the model
if __name__ == "__main__":

    # First injection after 3 years. This is to allow the model to reach a steady state before injection starts.
    dt_init = 3 * pp.YEAR
    
    schedule = np.array(
        [
            0.0,
            dt_init,
            dt_init + 1 * pp.YEAR,
            dt_init + 2 * pp.YEAR,
            dt_init + 3 * pp.YEAR,
            dt_init + 4 * pp.YEAR,  
            dt_init + 5 * pp.YEAR,
            dt_init + 6 * pp.YEAR
        ]
    ) 

    injection_well_fluxes = np.array([0.0, 0.0, 1e-6, 1.5e-6, 2e-6, 2.5e-6, 2.5e-6, 2.5e-6])  # m^3/month


    # If we want to just run firste time intervals, change length. 
    schedule_length = schedule.size
    schedule = schedule[:schedule_length]
    injection_well_fluxes = injection_well_fluxes[:schedule_length]
  
    # Define domain sizes, and fracture size. Depth: 4000m. Well 2000m deep. 
    length_scale = 1e3  # [m]
    domain_sizes = np.array( [ 5*length_scale, 0.5* length_scale])  # [m]


    grid_size = { "cell_size": length_scale/20, "cell_size_fracture": length_scale/4}
    stress = {"lithostatic_stress_multipliers": np.array([0.8, 1])}


    # sandstone
    sedimentary_values = {
                # Rock / poromechanics
                "biot_coefficient": 0.8,
                "density": 2400.0,                 # kg/m^3
                "porosity": 0.18,
                "permeability": 1e-14,             # m^2
                "specific_storage": 3e-10,       # Pa^-1

                # Elasticity
                "lame_lambda": 3.3e9,             # Pa
                "shear_modulus": 8.9e9,           # Pa

                # Friction / dilation
                "friction_coefficient": 0.7,
                "dilation_angle": 0.1,             # rad

                # Fracture mechanics / hydraulics
                # "fracture_normal_stiffness": 1.1e8,       # Pa/m
                # "fracture_tangential_stiffness": 5.0e7,   # Pa/m
                # "maximum_elastic_fracture_opening": 1e-3, # m
                "residual_aperture": 1e-3,                # m

                # Minumium gap
                "fracture_gap": 1e-4,                     # m
                "normal_permeability": 1.0e-10,           # m^2

                # Well model
                "well_radius": 0.1,                # m
                "skin_factor": 0.0}
    # granite
    crystalline_values = {
        
                "biot_coefficient": 0.47,
                "density": 2700.0,
                "porosity": 0.01,
                "permeability": 1e-18,
                "specific_storage": 1e-11,

                "lame_lambda": 1.5e10,
                "shear_modulus": 1.5e10,

                "friction_coefficient": 0.65,
                "dilation_angle": 0.10,

                # Fracture mechanics / hydraulics
                # "fracture_normal_stiffness": 1.1e8,       # Pa/m
                # "fracture_tangential_stiffness": 5.0e7,   # Pa/m
                # "maximum_elastic_fracture_opening": 1e-3, # m
                "residual_aperture": 1e-3,                # m

                # Minumium gap
                "fracture_gap": 1e-4,                     # m
                "normal_permeability": 1.0e-10,           # m^2

                # Well model
                "well_radius": 0.1,                # m
                "skin_factor": 0.0}

    model_params = {
        # Set time manager. Set to constant for now. 
        "time_manager": pp.TimeManager(
            schedule=schedule,
            dt_init=pp.YEAR/2,
            constant_dt=False,
            dt_min_max=(0.1 * pp.HOUR, pp.YEAR),
            iter_optimal_range=(6, 10),  # Allow more iterations than default.
            iter_relax_factors=(0.5, 1.8),  # More aggressive relaxation
        ),

        # Set physical parameters.
        **stress,
        "injection_well_fluxes": injection_well_fluxes,
            
        "material_constants": {
            "solid": pp.SolidConstants(**sedimentary_values),
            "fluid": pp.FluidComponent(**pp.fluid_values.water),  
            "numerical": pp.NumericalConstants(characteristic_displacement=1e-2),
            },
        "reference_variable_values": pp.ReferenceVariableValues(
        pressure=1e6
         ),  # type: ignore[arg-type]
        "units": pp.Units(m=1.0, kg=1.0e5, K=1.0),

            # Set geometry and meshing related parameters.
        "grid_type": "simplex",
        "meshing_arguments": grid_size,
        "domain_sizes": domain_sizes,
            # Dont know
        "adaptive_indicator_scaling": 1,
            # Set folder name for results.
        "folder_name": f"geothermal_reservoir2",
        }
    model = Simulation2D(model_params)

    # Dont care for now.
    solver_params = {
        "prepare_simulation": True,
        "nl_max_iterations": 25,  # Max iterations of a nonlinear solver (Newton)
        "nl_convergence_inc_atol": 1e-5,  # Increment norm
        "nl_convergence_res_atol": 1e-3,  # Residual norm
        "nl_divergence_inc_atol": 1e20,
        "nl_divergence_res_atol": 1e20,
        # Line search / Solution Strategies. These are considered "advanced" options,
        # improving the robustness of the nonlinear solver at the cost of some
        # additional computational overhead. Delete/comment the following lines for the
        # default Newton's method.
        "nonlinear_solver": line_search.ConstraintLineSearchNonlinearSolver,
        # Set to 1 to use turn on a residual-based line search. This involves some extra
        # residual evaluations and may be quite costly.
        "global_line_search": 0,
        # Set to 0 to use turn off the tailored line search, see the class
        # ConstraintLineSearchNonlinearSolver. This line search is cheap and has proven
        # effective for (some versions of) this particular simulation setup.
        "local_line_search": 1,
    }

    pp.run_time_dependent_model(model, solver_params)
