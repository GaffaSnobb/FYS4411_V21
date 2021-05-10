import os
import numpy as np

class Container:
    def __init__(
        self,
        data,
        method,
        n_particles,
        n_dims,
        n_mc_cycles,
        step_size,
        numerical,
        interaction,
        data_type,
        fname,
        a
    ):
        """
        Parameters
        ----------

        data : numpy.ndarray
            Array with data from data file.

        method : string
            'brute', 'importance', or 'gradient'.

        n_particles : integer
            Total number of particles.

        n_dims : integer
            Number of spatial dimensions.

        n_mc_cycles : integer
            Number of Monte Carlo cycles.

        step_size : float
            Either importance time step or brute force step size.

        numerical : boolean
            True if numerical differentiation, False if analytical.

        interaction : boolean
            Interaction term on / off.

        data_type : string
            'particles', 'onebody', or 'energies'.

        fname : string
            Original file name.

        a : float
            Interaction parameter.
        """
        self.data = data
        self.method = method
        self.n_particles = n_particles
        self.n_dims = n_dims
        self.n_mc_cycles = n_mc_cycles
        self.step_size = step_size
        self.numerical = numerical
        self.interaction = interaction
        self.data_type = data_type
        self.fname = fname
        self.a = a

def read_all_files(
    filter_method = None,
    filter_n_particles = None,
    filter_n_dims = None,
    filter_n_mc_cycles = None,
    filter_step_size = None,
    filter_numerical = None,
    filter_interaction = None,
    filter_data_type = None,
    filter_a = None,
    directory = "../src/generated_data/"
):
    """
    Read all text files in generated_data/ and store all relevant data
    in 'Container' objects. See Container docstring for details.

    Parameters
    ----------
    filter_method : NoneType, string
        Filter for only reading certain data files. Valid filters are
        'brute', 'importance', 'gradient'. If None, no filter is
        applied.

    filter_n_particles : NoneType, int
        Filter for choosing a certain amount of particles.

    filter_n_dims : NoneType, int
        Filter for choosing a certain number of dimensions.

    filter_n_mc_cycles : NoneType, int
        Filter for choosing a certain number of Monte Carlo cycles.

    filter_step_size : NoneType, float
        Filter for choosing a certain brute force or importance step
        size.

    filter_numerical : NoneType, boolean
        Filter for choosing numerical or analytical differentiation.

    filter_interaction : NoneType, boolean
        Filter for choosing interacting or non-interacting case.

    filter_data_type : NoneType, string
        Filter for only reading certain data types. Valid filters are
        'particle', 'onebody', 'energies'. If None, no filter is
        applied.

    filter_a : NoneType, float
        Filter for only reading a certain value for the interaction
        parameter.

    directory : string
        Location of data files. Path relative to the placement of this
        script.

    Returns
    -------
    data_list : list
        List of Container objects.
    """

    fnames = os.listdir(directory)
    data_list = []

    for i in range(len(fnames)):
        fname = fnames[i].split("_")
        if len(fname) < 10:
            """
            Skip 'README.md' (or other files which do not match the
            naming convention).
            """
            print(f"File {fnames[i]} skipped!")
            continue

        method = fname[1]
        if (filter_method != method) and (filter_method is not None):
            """
            Read only certain data files if filter_method is given as
            input.
            """
            continue

        n_particles = int(fname[2])
        if (filter_n_particles != n_particles) and (filter_n_particles is not None):
            """
            Read only certain data files if filter_n_particles is given as
            input.
            """
            continue

        n_dims = int(fname[3])
        if (filter_n_dims != n_dims) and (filter_n_dims is not None):
            """
            Read only certain data files if filter_n_dims is given as
            input.
            """
            continue

        n_mc_cycles = int(fname[4])
        if (filter_n_mc_cycles != n_mc_cycles) and (filter_n_mc_cycles is not None):
            """
            Read only certain data files if filter_n_mc_cycles is given as
            input.
            """
            continue

        step_size = float(fname[5])
        if (filter_step_size != step_size) and (filter_step_size is not None):
            """
            Read only certain data files if filter_step_size is given as
            input.
            """
            continue

        numerical = fname[6]
        if (filter_numerical and (numerical != "numerical")):
            continue

        elif ((not filter_numerical) and (numerical != "analytical")):
            continue

        if fname[6] == "numerical":
            numerical = True

        elif fname[6] == "analytical":
            numerical = False

        else:
            msg = f"Expected 'numerical' or 'analytical'. Got '{fname[6]}'."
            raise ValueError(msg)

        interaction = fname[7]
        if (filter_interaction and (interaction != "interaction") and (filter_interaction is not None)):
            continue

        elif ((not filter_interaction) and (interaction != "nointeraction") and (filter_interaction is not None)):
            print("LOL")
            continue

        if fname[7] == "interaction":
            interaction = True

        elif fname[7] == "nointeraction":
            interaction = False

        else:
            msg = f"Expected 'interaction' or 'nointeraction'. Got '{fname[7]}'."
            raise ValueError(msg)

        data_type = fname[8]
        if (filter_data_type != data_type) and (filter_data_type is not None):
            """
            Read only certain data files if filter_data_type is given as
            input.
            """
            continue

        try:
            a = float(fname[9])
        except ValueError:
            a = None    # For old data files without 'a' in the filename.

        if (filter_a != a) and (filter_a is not None) and (a is not None):
            continue

        data = np.loadtxt(fname = directory + fnames[i], skiprows=1)

        data_list.append(Container(
            data,
            method,
            n_particles,
            n_dims,
            n_mc_cycles,
            step_size,
            numerical,
            interaction,
            data_type,
            fnames[i],
            a
        ))

    if not data_list:
        msg = "No files have been loaded."
        msg += " Generate the data files or try another filter."
        msg += f"\n{filter_method=}"
        msg += f"\n{filter_n_particles=}"
        msg += f"\n{filter_n_dims=}"
        msg += f"\n{filter_n_mc_cycles=}"
        msg += f"\n{filter_step_size=}"
        msg += f"\n{filter_numerical=}"
        msg += f"\n{filter_interaction=}"
        msg += f"\n{filter_data_type=}"
        msg += f"\n{filter_a=}"
        msg += f"\n{directory=}"
        raise FileNotFoundError(msg)

    data_list.sort(key=lambda elem: elem.n_particles)   # Sort elements based on the number of particles.
    return data_list


if __name__ == "__main__":
    input = read_all_files(
        filter_method = "brute",
        filter_n_particles = 10,
        filter_n_dims = 3,
        filter_n_mc_cycles = 2**10,
        filter_step_size = 0.2,
        filter_numerical = False,
        filter_interaction = False,
        filter_data_type = "particles",
        filter_a = None,
        directory = "../src/generated_data/"
        )
