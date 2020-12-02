import numpy as np
import math
import sys
from mpi4py import MPI
from function import *
Nx=20;
Ny=20;
Nz=20;
asitecharge=1.93753;
bsitecharge=1.77706;
ositecharge=-1.23820;
filename='data.BFO';
cartlength=3.905;
celllength=cartlength*math.sqrt(2);
period=np.array([celllength*Nx,celllength/2.0*Ny,cartlength*Nz]);
comm=MPI.COMM_WORLD;
rank=comm.Get_rank();
if rank==0:
    [Asitep,Bsitep,Ositep]=writefile(filename,cartlength,Nx,Ny,Nz,asitecharge,bsitecharge,ositecharge);
    data=open(filename,'a');
    data.write("\n");
    data.write("Angles\n");
    data.write("\n");
else:
    Asitep=None;
    Bsitep=None;
    Ositep=None;
Asitep=comm.bcast(Asitep, root=0);
Bsitep=comm.bcast(Bsitep, root=0);
Ositep=comm.bcast(Ositep, root=0);
#specifying the angles:
#the first on is along z axis:
#Projection along [0,0,1]
oxygenlist=np.zeros([Nx,Ny,Nz]);
felist=np.zeros([Nx,Ny,Nz]);
size=comm.Get_size();
for k in range(rank,Nz,size):
    for j in range(Ny):
        for i in range(Nx):
            upoxygen=sortdistance(Bsitep[i+j*Nx+k*Nx*Ny], Ositep, period,3*Nx*Ny*Nz,np.array([0,0,1]));
            fe=sortdistance(Bsitep[i+j*Nx+k*Nx*Ny], Bsitep, period,Nx*Ny*Nz,np.array([0,0,1]));
            oxygenlist[i,j,k]=upoxygen;
            felist[i,j,k]=fe;
globaloxygen=np.zeros([Nx,Ny,Nz]);
globalfe=np.zeros([Nx,Ny,Nz]);
comm.Reduce(oxygenlist,globaloxygen,op=MPI.SUM,root=0);
comm.Reduce(felist,globalfe,op=MPI.SUM,root=0);
if rank==0:
    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                data.write(str(i+j*Nx+k*Nx*Ny+1)+" "+str(1)+" "+str(1+i+j*Nx+k*Nx*Ny+Nx*Ny*Nz)+" "+str(int(globaloxygen[i,j,k])+Nx*Ny*Nz*2+1)+" "+str(1+int(globalfe[i,j,k])+Nx*Ny*Nz)+"\n");
else:
    None;
for k in range(rank,Nz,size):
    for j in range(Ny):
        for i in range(Nx):
            upoxygen=sortdistance(Bsitep[i+j*Nx+k*Nx*Ny], Ositep, period,3*Nx*Ny*Nz,np.array([1,1,0]));
            fe=sortdistance(Bsitep[i+j*Nx+k*Nx*Ny], Bsitep, period,Nx*Ny*Nz,np.array([1,1,0]));
            oxygenlist[i,j,k]=upoxygen;
            felist[i,j,k]=fe;
globaloxygen=np.zeros([Nx,Ny,Nz]);
globalfe=np.zeros([Nx,Ny,Nz]);
comm.Reduce(oxygenlist,globaloxygen,op=MPI.SUM,root=0);
comm.Reduce(felist,globalfe,op=MPI.SUM,root=0);
if rank==0:
    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                data.write(str(i+j*Nx+k*Nx*Ny+1+Nx*Ny*Nz)+" "+str(1)+" "+str(1+i+j*Nx+k*Nx*Ny+Nx*Ny*Nz)+" "+str(int(globaloxygen[i,j,k])+Nx*Ny*Nz*2+1)+" "+str(1+int(globalfe[i,j,k])+Nx*Ny*Nz)+"\n");
else:
    None;
for k in range(rank,Nz,size):
    for j in range(Ny):
        for i in range(Nx):
            upoxygen=sortdistance(Bsitep[i+j*Nx+k*Nx*Ny], Ositep, period,3*Nx*Ny*Nz,np.array([1,-1,0]));
            fe=sortdistance(Bsitep[i+j*Nx+k*Nx*Ny], Bsitep, period,Nx*Ny*Nz,np.array([1,-1,0]));
            oxygenlist[i,j,k]=upoxygen;
            felist[i,j,k]=fe;
globaloxygen=np.zeros([Nx,Ny,Nz]);
globalfe=np.zeros([Nx,Ny,Nz]);
comm.Reduce(oxygenlist,globaloxygen,op=MPI.SUM,root=0);
comm.Reduce(felist,globalfe,op=MPI.SUM,root=0);
if rank==0:
    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                data.write(str(i+j*Nx+k*Nx*Ny+1+2*Nx*Ny*Nz)+" "+str(1)+" "+str(1+i+j*Nx+k*Nx*Ny+Nx*Ny*Nz)+" "+str(int(globaloxygen[i,j,k])+Nx*Ny*Nz*2+1)+" "+str(1+int(globalfe[i,j,k])+Nx*Ny*Nz)+"\n");
else:
    None;
if rank==0:
    data.close();
else:
    None;
