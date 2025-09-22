# MPS
A playground for algorithms related to matrix product states

## Usage 
Run script with `python mps_demo.py`

## Output
The script will generate a random state vector and convert it to a matrix product state (MPS) representation. It will then reconstruct the original state vector from the MPS and verify the fidelity of the reconstruction for a range of bond dimensions.
The fidelity is defined as the absolute square of the inner product between the original and reconstructed state vectors:
$F = |\langle \psi_0 | \phi_r \rangle|^2$
where $|\psi_0\rangle$ is the original state vector and $|\phi_r\rangle$ is the reconstructed state vector from the MPS.
