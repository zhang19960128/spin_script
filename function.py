import numpy as np
import math
import sys
from mpi4py import MPI
def writefile(filehandle,atomid,groupid,typeid,charge,position):
    filehandle.write(str(atomid)+" "+str(groupid)+" "+str(typeid)+" "+str(charge)+" "+str(position[0])+" "+str(position[1])+" "+str(position[2])+"\n");
def writefilespin(filehandle,atomid,groupid,typeid,charge,position,spinamp,spindirect):
    filehandle.write(str(atomid)+" "+str(groupid)+" "+str(typeid)+" "+str(charge)+" "+str(spinamp)+" "+str(position[0])+" "+str(position[1])+" "+str(position[2])+" "+str(spindirect[0])+" "+str(spindirect[1])+" "+str(spindirect[2])+"\n");
def changeindex(i,j,k,nx,ny,nz):
    return i+j*nx+k*nx*ny;
def sortdistance(origin,atomlist,period,length,direction):
    distance=np.zeros(length);
    localdist=np.zeros(3);
    for i in range(length):
        for k in range(3):
            localdist[k]=(atomlist[i,k]-origin[k])-round((atomlist[i,k]-origin[k])/period[k])*period[k];
            distance[i]=localdist[k]*localdist[k]+distance[i];
        distance[i]=math.sqrt(distance[i]);
    index=np.argsort(distance,kind='mergesort');
    chosenrange=6;
    prodlist=np.zeros(chosenrange);
    for i in range(chosenrange):
        for k in range(3):
            localdist[k]=(atomlist[index[i],k]-origin[k])-round((atomlist[index[i],k]-origin[k])/period[k])*period[k];
            prodlist[i]=prodlist[i]+localdist[k]*direction[k];
    indextwo=np.argsort(prodlist,kind='mergesort');
    return(index[indextwo[chosenrange-1]])
def writefile(filename,cartlength,Nx,Ny,Nz,asitecharge,bsitecharge,ositecharge):
    Asitep=np.zeros([Nx*Ny*Nz,3]);
    Bsitep=np.zeros([Nx*Ny*Nz,3]);
    Ositep=np.zeros([Nx*Ny*Nz*3,3]);
    celllength=cartlength*math.sqrt(2);
    data=open(filename,"w");
    data.write("#LAMMPS 100 water\n");
    data.write("\n");
    data.write(str(5*Nx*Ny*Nz)+" atoms\n");
    data.write(str(3*Nx*Ny*Nz)+" angles\n");
    data.write("3 atom types\n");
    data.write("1 angle types\n");
    data.write("\n");
    data.write("0.0 "+str(celllength*Nx)+" "+"xlo xhi\n");
    data.write("0.0 "+str(celllength/2.0*Ny)+" "+"ylo yhi\n");
    data.write("0.0 "+str(cartlength*Nz)+" "+"zlo zhi\n");
    data.write("\n");
    data.write("\n");
    data.write("Masses\n");
    data.write("\n")
    data.write('1 208.9804  #Bi\n')
    data.write('2  55.8450 #Fe\n')
    data.write('3  15.9994 #O\n')
    data.write('\n')
    data.write('Atoms\n')
    data.write('\n')
    reducefraction=1.0;
    #write the Asite atoms
    #creat the domains
    for k in range(Nz):# loop over z axis
        for j in range(Ny):#loop over y axis
            for i in range(Nx):# loop over x axis
                if(j%2==0):
                    if(i<Nx/2 and k>-1):
                        shift=[0.05*math.sqrt(2),0.0,-0.05];
                    elif(i>Nx/2 and k>-1):
                            shift=[0.05*math.sqrt(2),0.0,0.05];
    #            shift=[0.00,0.0,0.0]
                    if(k==0):
                        workcharge=asitecharge*reducefraction;
                        shift=[0.05*math.sqrt(2),0.0,0.00];
                    else:
                        workcharge=asitecharge;
                    position=np.array([i*celllength,j*celllength/2.0,k*cartlength])+np.array([celllength/2,0.0,0.0])+np.array(shift);
                    spinamp=0.0;
                    spindirect=np.array([0.0,0.0,0.0]);
                    Asitep[changeindex(i,j,k,Nx,Ny,Nz),0:3]=np.copy(position);
                    writefilespin(data,changeindex(i,j,k,Nx,Ny,Nz)+1,1,1,workcharge,position,spinamp,spindirect);
                else:
                    if(i<Nx/2 and k>-1):
                        shift=[0.05*math.sqrt(2),0.0,-0.05];
                    elif(i>Nx/2 and k>-1):
                        shift=[0.05*math.sqrt(2),0.0,0.05];
    #        shift=[0.00,0.0,0.0];
                    if(k==0):
                        workcharge=asitecharge*reducefraction;
                        shift=[0.05*math.sqrt(2),0.0,0.00];
                    else:
                        workcharge=asitecharge;
                    position=np.array([i*celllength,j*celllength/2.0,k*cartlength])+np.array(shift);
                    spinamp=0.0;
                    spindirect=np.array([0.0,0.0,0.0]);
                    Asitep[changeindex(i,j,k,Nx,Ny,Nz),0:3]=np.copy(position);
                    writefilespin(data,changeindex(i,j,k,Nx,Ny,Nz)+1,1,1,workcharge,position,spinamp,spindirect);
    #write the Bsite atoms
    spindirect=np.array([0,0,1]);
    for k in range(Nz):#loop over z axis
        for j in range(Ny):#loop over y axis
            for i in range(Nx):#loop over x aixs
                if(k==Nz-1):
                    workcharge=bsitecharge*reducefraction;
                else:
                    workcharge=bsitecharge;
                if(j%2==0):
                    shift=[0.0,0.0,cartlength/2.0];
                    position=np.array([i*celllength,j*celllength/2.0,k*cartlength])+np.array(shift);
                    spinamp=4.0;
                    if k%2==0:
                        spindir=1*spindirect;
                    else:
                        spindir=-1*spindirect;
                    if j%2==0:
                        spindir=1*spindir;
                    else:
                        spindir=-1*spindir;
                    Bsitep[changeindex(i,j,k,Nx,Ny,Nz),0:3]=np.copy(position);
                    writefilespin(data,changeindex(i,j,k,Nx,Ny,Nz)+Nx*Ny*Nz+1,1,2,workcharge,position,spinamp,spindir);
                else:
                    if k%2==0:
                        spindir=1*spindirect;
                    else:
                        spindir=-1*spindirect;
                    if j%2==0:
                        spindir=1*spindir;
                    else:
                        spindir=-1*spindir;
                    shift=[celllength/2.0,0.0,cartlength/2.0];
                    spinamp=4.0;
                    position=np.array([i*celllength,j*celllength/2.0,k*cartlength])+np.array(shift);
                    Bsitep[changeindex(i,j,k,Nx,Ny,Nz),0:3]=np.copy(position);
                    writefilespin(data,changeindex(i,j,k,Nx,Ny,Nz)+Nx*Ny*Nz+1,1,2,workcharge,position,spinamp,spindir);
    #wGGrite the O1 atoms
    # the down part of the atoms
    #reference to Fe atoms
    shiftO1=np.array([0.0,0.0,-1.0*cartlength/2.0]);
    shiftO2=np.array([math.sqrt(2)/4.0*cartlength,-1*math.sqrt(2)/4.0*cartlength,0]);
    shiftO3=np.array([math.sqrt(2)/4.0*cartlength,math.sqrt(2)/4.0*cartlength,0]);
    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                if k==0:
                    workcharge=ositecharge*reducefraction;
                else:
                    workcharge=ositecharge;
                if(j%2==0):
                    reference=np.array([i*celllength,j*celllength/2.0,k*cartlength])+np.array([0.0,0.0,cartlength/2.0]);
                    spinamp=0.0;
                    spindirect=np.array([0.0,0.0,0.0]);
                    Ositep[changeindex(i,j,k,Nx,Ny,Nz),0:3]=reference+shiftO1;
                    writefilespin(data,changeindex(i,j,k,Nx,Ny,Nz)+2*Nx*Ny*Nz+1,1,3,workcharge,reference+shiftO1,spinamp,spindirect);
                else:
                    reference=np.array([i*celllength,j*celllength/2.0,k*cartlength])+np.array([celllength/2.0,0.0,cartlength/2.0]);
                    Ositep[changeindex(i,j,k,Nx,Ny,Nz),0:3]=reference+shiftO1;
                    writefilespin(data,changeindex(i,j,k,Nx,Ny,Nz)+2*Nx*Ny*Nz+1,1,3,workcharge,reference+shiftO1,spinamp,spindirect);
    # the right part of the atoms
    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                if k==Nz-1:
                    workcharge=ositecharge*reducefraction;
                else:
                    workcharge=ositecharge;
                if(j%2==0):
                    reference=np.array([i*celllength,j*celllength/2.0,k*cartlength])+np.array([0.0,0.0,cartlength/2.0]);
                    spinamp=0.0;
                    spindirect=np.array([0.0,0.0,0.0]);
                    Ositep[changeindex(i,j,k,Nx,Ny,Nz)+1*Nx*Ny*Nz,0:3]=reference+shiftO2;
                    writefilespin(data,changeindex(i,j,k,Nx,Ny,Nz)+3*Nx*Ny*Nz+1,1,3,workcharge,reference+shiftO2,spinamp,spindirect);
                else:
                    spinamp=0.0;
                    spindirect=np.array([0.0,0.0,0.0]);
                    reference=np.array([i*celllength,j*celllength/2.0,k*cartlength])+np.array([celllength/2.0,0.0,cartlength/2.0]);
                    Ositep[changeindex(i,j,k,Nx,Ny,Nz)+1*Nx*Ny*Nz,0:3]=reference+shiftO2;
                    writefilespin(data,changeindex(i,j,k,Nx,Ny,Nz)+3*Nx*Ny*Nz+1,1,3,workcharge,reference+shiftO2,spinamp,spindirect);
    # the up part of the atoms
    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                if k==Nz-1:
                    workcharge=ositecharge*reducefraction;
                else:
                    workcharge=ositecharge;
                if(j%2==0):
                    reference=np.array([i*celllength,j*celllength/2.0,k*cartlength])+np.array([0.0,0.0,cartlength/2.0]);
                    spinamp=0.0;
                    spindirect=np.array([0.0,0.0,0.0]);
                    Ositep[changeindex(i,j,k,Nx,Ny,Nz)+2*Nx*Ny*Nz,0:3]=reference+shiftO3;
                    writefilespin(data,changeindex(i,j,k,Nx,Ny,Nz)+4*Nx*Ny*Nz+1,1,3,workcharge,reference+shiftO3,spinamp,spindirect);
                else:
                    reference=np.array([i*celllength,j*celllength/2.0,k*cartlength])+np.array([celllength/2.0,0.0,cartlength/2.0]);
                    spinamp=0.0;
                    spindirect=np.array([0.0,0.0,0.0]);
                    Ositep[changeindex(i,j,k,Nx,Ny,Nz)+2*Nx*Ny*Nz,0:3]=reference+shiftO3;
                    writefilespin(data,changeindex(i,j,k,Nx,Ny,Nz)+4*Nx*Ny*Nz+1,1,3,workcharge,reference+shiftO3,spinamp,spindirect);
    return [Asitep,Bsitep,Ositep];

