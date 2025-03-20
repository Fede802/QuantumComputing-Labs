#Imports
import matplotlib.pyplot as plt  # for plots
import numpy as np
import qiskit
from qiskit.quantum_info import Operator  # to extract the circuit matrix
from qiskit.visualization import array_to_latex # to display matrices and state vectors in latex format
from qiskit.visualization import plot_bloch_vector, plot_bloch_multivector,plot_state_qsphere # to display results in a nice way
import math
import sys
from IPython.display import display, Markdown, Latex

# get the Qiskit version
def get_qiskit_version():
    print("The current qiskit version is: " + qiskit.__version__)  

# get the python version
def get_python_version():
    print("The current python version is: " + sys.version)

# conversion from statevector to spherical coordinates
def get_spherical_coordinates(statevector):
    # Convert to polar form:
    r0 = np.abs(statevector[0])
    ϕ0 = np.angle(statevector[0])

    r1 = np.abs(statevector[1])
    ϕ1 = np.angle(statevector[1])

    # Calculate the coordinates:
    r = np.sqrt(r0 ** 2 + r1 ** 2)
    θ = 2 * np.arccos(r0 / r)
    ϕ = ϕ1 - ϕ0
    
    return [r,θ,ϕ]

# Example 
# Ψ = [complex(1 / np.sqrt(2), 0), complex(1 / np.sqrt(2), 0)]
# plot_bloch_vector(get_spherical_coordinates(Ψ), coord_type = 'spherical')

# conversion from state vector to cartesian coordinates
def get_cartesian_coordinates(statevector):
    # convert to polar form
    [r, θ, ϕ] = get_spherical_coordinates(statevector);
    # since the radius of Bloch sphere is 1 we set r=1
    r=1
    x = r*math.sin(θ)*math.cos(ϕ)
    y = r*math.sin(θ)*math.sin(ϕ)
    z = r*math.cos(θ)
    
    return [x,y,z]
#Example
#[x,y,z]=get_cartesian_coordinates(final_state)
#print(x,y,z)

def plot_two_statevectors_on_bloch_sphere(statevectorIn, statevectorOut, coord_type="spherical"):
    # plot the two state vectors of a single qubit circuit on the Bloch sphere
    """Args:
        statevectorIn: The input state vector.
        statevectorOut  The output state vector.
    Returns: A plot of the two state vectors on the Bloch sphere.
    """ 
    if coord_type == "spherical": # convert statevectors to spherical coordinates
        coord_in = get_spherical_coordinates(statevectorIn)
        coord_out = get_spherical_coordinates(statevectorOut)    
    
    else: # convert statevectors to cartesian coordinates
        coord_in = get_cartesian_coordinates(statevectorIn)
        coord_out = get_cartesian_coordinates(statevectorOut)
    
    # plot the two state vectors
    fig = plt.figure(figsize = [6, 9])

    states = [
        coord_in, # input state vector
        coord_out, # output state vector
        ]
    
        # Values are in fractions of figure width and height:
    positions = [
        [0, 0],
        [0.5, 0],
        ]
    titles=['Input state vector','Output state vector']
    for i in range(2):        
        ax = fig.add_axes([positions[i][0], positions[i][1], 0.5, 0.333], projection='3d')
        plot_bloch_vector(states[i],  coord_type=coord_type, title = titles[i], ax = ax)
    plt.show()

def plot_statevector(stateVec,label):
    """
    Plots the real and imaginary parts of a state vector in a bar chart.
    Args: 
      stateVec: the state vector to be plotted
      label: the title of the plot
      Returns:
      A bar chart with the real and imaginary parts of the state vector.
     """
    # Get the dimension of the state vector
    dim=stateVec.data.shape[0]
    # Create the xticks and xticks_labels
    my_xticks=[i for i in range(dim)]
    my_xticks_labels = [format(i, '0'+str(int(np.log2(dim)))+'b') for i in range(dim)]

    stateVecAL = np.array(stateVec)
    fig, axs = plt.subplots(2)
    fig.suptitle(label)
    markerline0, stemlines0, baseline0 = axs[0].stem(
        np.arange(0, dim, 1), stateVecAL.real, 'tab:blue')
    axs[0].set(ylabel='Real Part', xticks=my_xticks)
    markerline1, stemlines1, baseline1 = axs[1].stem(
        np.arange(0, dim, 1), stateVecAL.imag, 'tab:orange')
    axs[1].set(xlabel='States', ylabel='Imaginary Part')
    plt.setp(axs, xticks=np.arange(0, dim, 1), xticklabels=my_xticks_labels)
    plt.setp(baseline0, 'color', 'k', 'linewidth', 2)
    plt.setp(baseline1, 'color', 'k', 'linewidth', 2)
    plt.show()
    
def derive_unitary_matrix(circuit):
    """
    Derive the unitary matrix from a quantum circuit.    
    Args:   
        circuit (QuantumCircuit): The quantum circuit to derive the unitary matrix from. 
        Returns: The unitary matrix of the quantum circuit in latex format.  
    """
    # Define the unitary operator from the circuit
    UnitaryRepresentation= Operator(circuit)
    # Print the matrix
    Unitary_latex= array_to_latex(UnitaryRepresentation.data)
    return Unitary_latex

def print_unitary(circuit1, circuit2):
    """
    Print the unitary matrix of two quantum circuit in latex format.

    Parameters:
    circuit1 (QuantumCircuit): The first quantum circuit to print the unitary matrix of.
    circuit2 (QuantumCircuit): The second quantum circuit to print the unitary matrix of.
    """
    # Convert the circuit to a unitary matrix and print it
    display(array_to_latex(Operator(circuit1).data),array_to_latex(Operator(circuit2).data))

def compare_unitary(circuit1, circuit2):
    """
    Compare two quantum circuits by converting them to unitary matrices and checking if they are equivalent.

    Parameters:
    circuit1 (QuantumCircuit): The first quantum circuit to compare.
    circuit2 (QuantumCircuit): The second quantum circuit to compare.

    Returns:
    bool: True if the unitary matrices of the circuits are equivalent, False otherwise.
    """

    # Convert the circuits to unitary matrices
    unitary1 = Operator(circuit1)
    unitary2 = Operator(circuit2)
    print_unitary(circuit1, circuit2)
    
    # Compare the unitary matrices
    if (unitary1.equiv(unitary2)):
        return True
    else:
        return False

def evolve_state_vector(circuit, initial_state_vector):
    """Derive the state vector from a quantum circuit.    
    Args:   
        circuit (QuantumCircuit): The quantum circuit to derive the state vector from. 
        initial_state_vector (array): The initial state vector to evolve. 
        Returns: The state vector of the quantum circuit in latex format.  
    """
   # Evolve the state vector using the given circuit
    statevectorOut = initial_state_vector.evolve(circuit)
    # Display the evolved state vector
    statevectorOut_latex=array_to_latex(statevectorOut, prefix="\\text{Statevector = }")
    return statevectorOut_latex

def display_state_vector(h, sv, isv):
    display(Markdown("### "+h))
    display(Latex(array_to_latex(sv, prefix="\\text{Statevector = }").data[1:][:-2] + " = \\displaystyle"+ isv +"$"))

def display_io_info(isv, isvc, osv, osvc):
    display_state_vector("Input Statevector", isv, isvc)
    display_state_vector("Output Statevector", osv, osvc)

def display_structure(qc):
    display(Markdown("### Layout"))
    display(qc.draw(output='mpl'))
    display(Markdown("### Unitary Matrix"))
    display(Latex("$"+derive_unitary_matrix(qc).data[4:][:-3]+"$"))
    
def display_info(qc, isv, isvc, osv, osvc):
    display_structure(qc)
    display_io_info(isv, isvc, osv, osvc)
