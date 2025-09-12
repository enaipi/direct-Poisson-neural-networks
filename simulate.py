import argparse
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

from models.RigidBody import *
import pandas as pd
from learn import DEFAULT_folder_name

def save_simulation(data_frame, file_name): #save data to file_name
    """
    The function `save_simulation` saves a data frame to a file with the specified file name.
    
    :param data_frame: A pandas DataFrame containing the simulation data that you want to save to a file
    :param file_name: The name of the file where the data will be saved
    """
    print("Storing to: ", file_name, "\n")
    data_frame.to_csv(file_name)

# not used
def simulate(args, method = "normal"): #simulate with args given below 
    """
    Simulation of a given model.

    :param args: Parameters passed from the command line, see the main method.
    :param method: Method of simulation (normal means using a traditional integrator, other methods use neural networks)
    """
    #Create d2E matrix
    d2E = np.array([[1/args.Ix,0,0],\
                    [0,1/args.Iy,0],\
                    [0,0, 1/args.Iz]])
                    
    ts, ms, msqs, Ls, Es, casss, rs, mrs, rsqs = [], [], [], [], [], [], [], [], []

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda  and args.simulate_cuda else "cpu")

    if args.model == "RB": # Rigid body
        if method == "implicit":
            if args.scheme == "IMR":
                solver = RBNeuralIMR(args.Ix,args.Iy,args.Iz, d2E, args.init_mx,args.init_my,args.init_mz,args.dt,args.alpha, method = "implicit", name = args.folder_name, device=device)
            else:
                solver = Neural(args.Ix,args.Iy,args.Iz, d2E, args.init_mx,args.init_my,args.init_mz,args.dt,args.alpha, method = "implicit", name = args.folder_name, device=device)
        elif method == "soft":
            if args.scheme == "IMR":
                solver = RBNeuralIMR(args.Ix,args.Iy,args.Iz, d2E, args.init_mx,args.init_my,args.init_mz,args.dt,args.alpha, method = "soft", name = args.folder_name, device=device)
            else:
                solver = Neural(args.Ix,args.Iy,args.Iz, d2E, args.init_mx,args.init_my,args.init_mz,args.dt,args.alpha, method = "soft", name = args.folder_name, device=device)
        elif method == "without":
            if args.scheme == "IMR":
                solver = RBNeuralIMR(args.Ix,args.Iy,args.Iz, d2E, args.init_mx,args.init_my,args.init_mz,args.dt,args.alpha, method = "without", name = args.folder_name, device=device)
            else:
                solver = Neural(args.Ix,args.Iy,args.Iz, d2E, args.init_mx,args.init_my,args.init_mz,args.dt,args.alpha, method = "without", name = args.folder_name, device=device)
        elif method =="normal":
            if args.scheme == "Eh":
                solver = RBEhrenfest(args.Ix,args.Iy,args.Iz, d2E, args.init_mx,args.init_my,args.init_mz,args.dt,args.alpha)
            elif args.scheme == "CN":
                solver = RBESeReCN(args.Ix,args.Iy,args.Iz, d2E, args.init_mx,args.init_my,args.init_mz,args.dt,args.alpha)
            elif args.scheme == "FE":
                solver = RBESeReFE(args.Ix,args.Iy,args.Iz, d2E, args.init_mx,args.init_my,args.init_mz,args.dt,args.alpha)
            elif args.scheme == "IMR":
                solver = RBIMR(args.Ix,args.Iy,args.Iz, d2E, args.init_mx,args.init_my,args.init_mz,args.dt)
            elif args.scheme == "RK4": 
                solver = RBRK4(args.Ix,args.Iy,args.Iz, d2E, args.init_mx,args.init_my,args.init_mz,args.dt,args.M_tau)
            else:
                raise Exception("Unknown scheme.")
        else:
            raise Exception("Unknown method.")

    elif args.model == "HT": # Heavy top
        if method == "implicit":
            raise Exception("Implicit solver not yet implemented for HT.")
            #solver = NeuralHT(args.Ix,args.Iy,args.Iz, d2E, args.init_mx,args.init_my,args.init_mz,args.dt,args.alpha, method = "implicit")
        elif method == "soft":
            if args.scheme == "IMR":
                solver = HeavyTopNeuralIMR(args.Ix,args.Iy,args.Iz, d2E, args.init_mx,args.init_my,args.init_mz,args.dt,args.alpha, args.Mgl, args.init_rx, args.init_ry, args.init_rz, method = "soft", name = args.folder_name)
            else:
                solver = HeavyTopNeural(args.Ix,args.Iy,args.Iz, d2E, args.init_mx,args.init_my,args.init_mz,args.dt,args.alpha, args.Mgl, args.init_rx, args.init_ry, args.init_rz, method = "soft", name = args.folder_name)
        elif method == "without":
            if args.scheme == "IMR":
                solver = HeavyTopNeuralIMR(args.Ix,args.Iy,args.Iz, d2E, args.init_mx,args.init_my,args.init_mz,args.dt,args.alpha, args.Mgl, args.init_rx, args.init_ry, args.init_rz, method = "without", name = args.folder_name)
            else:
                solver = HeavyTopNeural(args.Ix,args.Iy,args.Iz, d2E, args.init_mx,args.init_my,args.init_mz,args.dt,args.alpha, args.Mgl, args.init_rx, args.init_ry, args.init_rz, method = "without", name = args.folder_name)
        elif method =="normal":
            #if args.scheme == "Eh":
            #    solver = RBEhrenfest(args.Ix,args.Iy,args.Iz, d2E, args.init_mx,args.init_my,args.init_mz,args.dt,args.alpha)
            #elif args.scheme == "CN":
            if args.scheme == "IMR":
                solver = HeavyTopIMR(args.Ix,args.Iy,args.Iz, d2E, args.init_mx,args.init_my,args.init_mz,args.dt,args.alpha, args.Mgl, args.init_rx, args.init_ry, args.init_rz)
            else:
                solver = HeavyTopCN(args.Ix,args.Iy,args.Iz, d2E, args.init_mx,args.init_my,args.init_mz,args.dt,args.alpha, args.Mgl, args.init_rx, args.init_ry, args.init_rz)
            #elif args.scheme == "FE":
            #    solver = RBESeReFE(args.Ix,args.Iy,args.Iz, d2E, args.init_mx,args.init_my,args.init_mz,args.dt,args.alpha)
            #else:
            #    raise Exception("Unknown scheme.")
        else:
            raise Exception("Unknown method.")
    elif args.model == "P3D" or args.model == "K3D": # Particle in 3D or Kepler problem in 3D (not tested yet)
        if method == "implicit":
            raise Exception("Implicit solver not yet implemented for P3D.")
            #solver = NeuralHT(args.Ix,args.Iy,args.Iz, d2E, args.init_mx,args.init_my,args.init_mz,args.dt,args.alpha, method = "implicit")
        elif method == "soft":
            if args.scheme == "IMR":
                solver = Particle3DNeuralIMR(args.M, args.dt, args.alpha, args.init_rx, args.init_ry, args.init_rz, args.init_mx,args.init_my,args.init_mz, method = "soft", name = args.folder_name)
            else:
                solver = Particle3DNeural(args.M, args.dt, args.alpha, args.init_rx, args.init_ry, args.init_rz, args.init_mx,args.init_my,args.init_mz, method = "soft", name = args.folder_name)
        elif method == "without":
            if args.scheme == "IMR":
                solver = Particle3DNeuralIMR(args.M, args.dt, args.alpha, args.init_rx, args.init_ry, args.init_rz, args.init_mx,args.init_my,args.init_mz, method = "without", name = args.folder_name)
            else:
                solver = Particle3DNeural(args.M, args.dt, args.alpha, args.init_rx, args.init_ry, args.init_rz, args.init_mx,args.init_my,args.init_mz, method = "without", name = args.folder_name)
        elif method =="normal":
            #if args.scheme == "Eh":
            #    solver = RBEhrenfest(args.Ix,args.Iy,args.Iz, d2E, args.init_mx,args.init_my,args.init_mz,args.dt,args.alpha)
            #elif args.scheme == "CN":
            if args.scheme == "IMR":
                if args.model == "K3D":
                    solver = Particle3DKeplerIMR(args.M, args.dt, args.alpha, args.init_rx, args.init_ry, args.init_rz, args.init_mx,args.init_my,args.init_mz)
                else:
                    solver = Particle3DIMR(args.M, args.dt, args.alpha, args.init_rx, args.init_ry, args.init_rz, args.init_mx,args.init_my,args.init_mz)
            else:
                if args.model == "K3D":
                    raise Exception("Don't use CN for Kepler.")
                solver = Particle3DCN(args.M, args.dt, args.alpha, args.init_rx, args.init_ry, args.init_rz, args.init_mx,args.init_my,args.init_mz)
            #elif args.scheme == "FE":
            #    solver = RBESeReFE(args.Ix,args.Iy,args.Iz, d2E, args.init_mx,args.init_my,args.init_mz,args.dt,args.alpha)
            #else:
            #    raise Exception("Unknown scheme.")
    elif args.model == "P2D": # Particle in 2D
        if method == "implicit":
            raise Exception("Implicit solver not yet implemented for P2D.")
            #solver = NeuralHT(args.Ix,args.Iy,args.Iz, d2E, args.init_mx,args.init_my,args.init_mz,args.dt,args.alpha, method = "implicit")
        elif method == "soft":
            if args.scheme == "IMR":
                solver = Particle2DNeural(args.M, args.dt, args.alpha, args.init_rx, args.init_ry, args.init_mx,args.init_my, zeta = args.zeta, method = "soft", name = args.folder_name)
            else:
                raise Exception("Not implemented.")
                #solver = Particle3DNeural(args.M, args.dt, args.alpha, args.init_rx, args.init_ry, args.init_rz, args.init_mx,args.init_my,args.init_mz, method = "soft", name = args.folder_name)
        elif method == "without":
            if args.scheme == "IMR":
                solver = Particle2DNeural(args.M, args.dt, args.alpha, args.init_rx, args.init_ry, args.init_mx,args.init_my, zeta = args.zeta,method = "without", name = args.folder_name)
            else:
                raise Exception("Not implemented.")
                #solver = Particle3DNeural(args.M, args.dt, args.alpha, args.init_rx, args.init_ry, args.init_rz, args.init_mx,args.init_my,args.init_mz, method = "without", name = args.folder_name)
        elif method =="normal":
            #if args.scheme == "Eh":
            #    solver = RBEhrenfest(args.Ix,args.Iy,args.Iz, d2E, args.init_mx,args.init_my,args.init_mz,args.dt,args.alpha)
            #elif args.scheme == "CN":
            if args.scheme == "IMR":
                solver = Particle2DIMR(args.M, args.dt, args.alpha, args.init_rx, args.init_ry, args.init_mx,args.init_my,args.zeta)
            else:
                raise Exception("Not implemented.")
                #solver = Particle3DCN(args.M, args.dt, args.alpha, args.init_rx, args.init_ry, args.init_rz, args.init_mx,args.init_my,args.init_mz)
        else:
            raise Exception("Unknown method.")
    elif args.model == "Sh": # Shivamoggi equations
        if abs(args.init_ry)>0.1: #empirically unstable
            args.init_ry *= 1/(10*abs(args.init_ry))
        if method == "implicit":
            raise Exception("Implicit solver not yet implemented for Shivamoggi.")
            #solver = NeuralHT(args.Ix,args.Iy,args.Iz, d2E, args.init_mx,args.init_my,args.init_mz,args.dt,args.alpha, method = "implicit")
        elif method == "soft":
            if args.scheme == "IMR":
                solver = ShivamoggiNeural(args.M, args.dt, args.alpha, args.init_rx, args.init_ry, args.init_rz,args.init_mx, method = "soft", name = args.folder_name)
            else:
                raise Exception("Not implemented.")
                #solver = Particle3DNeural(args.M, args.dt, args.alpha, args.init_rx, args.init_ry, args.init_rz, args.init_mx,args.init_my,args.init_mz, method = "soft", name = args.folder_name)
        elif method == "without":
            if args.scheme == "IMR":
                solver = ShivamoggiNeural(args.M, args.dt, args.alpha, args.init_rx, args.init_ry, args.init_rz,args.init_mx, method = "without", name = args.folder_name)
            else:
                raise Exception("Not implemented.")
                #solver = Particle3DNeural(args.M, args.dt, args.alpha, args.init_rx, args.init_ry, args.init_rz, args.init_mx,args.init_my,args.init_mz, method = "without", name = args.folder_name)
        elif method =="normal":
            #if args.scheme == "Eh":
            #    solver = RBEhrenfest(args.Ix,args.Iy,args.Iz, d2E, args.init_mx,args.init_my,args.init_mz,args.dt,args.alpha)
            #elif args.scheme == "CN":
            if args.scheme == "IMR":
                solver = ShivamoggiIMR(args.M, args.dt, args.alpha, args.init_rx, args.init_ry, args.init_rz,args.init_mx)
            else:
                raise Exception("Not implemented.")
                #solver = Particle3DCN(args.M, args.dt, args.alpha, args.init_rx, args.init_ry, args.init_rz, args.init_mx,args.init_my,args.init_mz)
        else:
            raise Exception("Unknown method.")

    #Timesteps
    dt = args.dt

    #Preparing files for output
    if args.model == "RB":
        m = np.array([args.init_mx,args.init_my,args.init_mz])
        r = np.array([args.init_rx,args.init_ry,args.init_rz]) #not used. Just not to cause errors
        Ls.append(solver.get_L(m))
        Es.append(solver.get_E(m))
    elif args.model == "HT":
        m = np.array([args.init_mx,args.init_my,args.init_mz])
        r = np.array([args.init_rx,args.init_ry,args.init_rz])
        mr = np.concatenate([m,r])
        Ls.append(solver.get_L(mr))
        Es.append(solver.get_E(mr))
    elif args.model == "P3D" or args.model == "K3D":
        m = np.array([args.init_mx,args.init_my,args.init_mz])
        r = np.array([args.init_rx,args.init_ry,args.init_rz])
        mr = np.concatenate([r,m])
        Ls.append(solver.get_L(mr))
        Es.append(solver.get_E(mr))
    elif args.model == "P2D":
        m = np.array([args.init_mx,args.init_my])
        r = np.array([args.init_rx,args.init_ry])
        mr = np.concatenate([r,m])
        Ls.append(solver.get_L(mr))
        Es.append(solver.get_E(mr))
    elif args.model == "Sh":
        r = np.array([args.init_mx, args.init_rx,args.init_ry, args.init_rz]) #not used
        m = np.array([args.init_mx, args.init_rx,args.init_ry, args.init_rz])
        Ls.append(solver.get_L(m))
        Es.append(solver.get_E(m))
    else:
        raise Exception("Unknown model.")
    
    msq = m*m
    msqs.append(msq)
    ts.append(0)
    ms.append(m)
    rs.append(r)
    mrs.append(np.dot(m,r))
    rsqs.append(np.dot(r,r))
    if method == "implicit":
        casss.append(solver.get_cass(m)) #not yet implemented for HT
    else:
        casss.append(0.0) #not implemented yet

    store_each = 1

    #calculate evolution
    for i in range(args.steps-1):
        #print(i)
        m_old = np.array(m, copy=True)
        r_old = np.array(r, copy=True)
        if args.model == "RB" or args.model == "Sh":
            m = solver.m_new()
        elif args.model == "HT":
            (m, r) = solver.m_new()
        elif args.model == "P3D" or args.model == "K3D" or args.model=="P2D":
            (r, m) = solver.m_new()

        if i % store_each == 0:
            t = dt * i
            msq = np.dot(m,m)
            ts.append(t)
            ms.append(m)
            msqs.append(msq)
            if args.model == "RB" or args.model == "Sh":
                Ls.append(solver.get_L(m))
                Es.append(solver.get_E(m))
            elif args.model == "HT":
                mr = np.concatenate([m,r])
                Ls.append(solver.get_L(mr))
                Es.append(solver.get_E(mr))
                rs.append(np.array(r, copy=True))
                mrs.append(np.dot(m,r))
                rsqs.append(np.dot(r,r))
            elif args.model == "P3D" or args.model == "K3D":
                Ls.append(solver.get_L(mr))
                Es.append(solver.get_E(mr))
                rs.append(np.array(r, copy=True))
                rsqs.append(np.dot(r,r))
            elif args.model == "P2D":
                Ls.append(solver.get_L(mr))
                Es.append(solver.get_E(mr))
                rs.append(np.array(r, copy=True))
                rsqs.append(np.dot(r,r))
            if method == "implicit":
                casss.append(solver.get_cass(m))
            else:
                casss.append(0.0) #Casimirs implemented only in the implicit case, so far

        if args.verbose and (i % (max(1,int(round(args.steps/100)))) == 0):
            print(i/args.steps*100, "%")
            print("|m| = ", np.linalg.norm(m))
            print("E = ", 0.5 * m @ d2E @ m)
    
    ts = np.array(ts[1:]).reshape(-1,1)

    ms_old = np.array(ms[:-1])
    ms = np.array(ms[1:])
    msqs = np.array(msqs[1:]).reshape(-1,1)
    Ls = np.array(Ls[1:])
    Es = np.array(Es[1:]).reshape(-1,1)
    casss = np.array(casss[1:]).reshape(-1,1)

    rs_old = np.array(rs[:-1])
    rs = np.array(rs[1:])
    mrs = np.array(mrs[1:]).reshape(-1,1)
    rsqs = np.array(rsqs[1:]).reshape(-1,1)
    
    # won't use
    """else:
        results = run_simulation_gpu(solver, m, r, args.model, args.steps)

        ts = results["ts"][1:].reshape(-1, 1)
        ms_old = results["ms"][:-1]
        ms = results["ms"][1:]
        msqs = results["msqs"][1:].reshape(-1, 1)
        Ls = results["Ls"][1:]
        Es = results["Es"][1:].reshape(-1, 1)
        casss = results["casss"][1:].reshape(-1, 1)
        if results["rs"] is not None:
            rs_old = results["rs"][:-1]
            rs = results["rs"][1:]
            mrs = results["mrs"][1:].reshape(-1, 1)
            rsqs = results["rsqs"][1:].reshape(-1, 1)"""

    if method == "implicit": #returns also the Casimir
        data = np.hstack((ts, ms_old, ms, msqs, Ls[:,0,1].reshape(-1,1), Ls[:,0,2].reshape(-1,1), Ls[:,1,2].reshape(-1,1), Es, casss))
        return pd.DataFrame(data, columns = ["time", "old_mx", "old_my", "old_mz", "mx", "my", "mz", "sqm", "L_01", "L_02", "L_12", "E", "cass"])#return results of simulation
    else:
        if args.model == "RB":
            data = np.hstack((ts, ms_old, ms, msqs, Ls[:,0,1].reshape(-1,1), Ls[:,0,2].reshape(-1,1), Ls[:,1,2].reshape(-1,1), Es))
            return pd.DataFrame(data, columns = ["time", "old_mx", "old_my", "old_mz", "mx", "my", "mz", "sqm", "L_01", "L_02", "L_12", "E"])#return results of simulation
        elif args.model =="HT":
            data = np.hstack((ts, ms_old, rs_old, ms, rs, msqs, Ls[:,0,1].reshape(-1,1), Ls[:,0,2].reshape(-1,1), Ls[:,0,3].reshape(-1,1), Ls[:,0,4].reshape(-1,1), Ls[:,0,5].reshape(-1,1), Ls[:,1,2].reshape(-1,1), Ls[:,1,3].reshape(-1,1), Ls[:,1,4].reshape(-1,1), Ls[:,1,5].reshape(-1,1), Ls[:,2,3].reshape(-1,1), Ls[:,2,4].reshape(-1,1), Ls[:,2,5].reshape(-1,1), Ls[:,3,4].reshape(-1,1), Ls[:,3,5].reshape(-1,1), Ls[:,4,5].reshape(-1,1), Es, mrs, rsqs))
            return pd.DataFrame(data, columns = ["time", "old_mx", "old_my", "old_mz", "old_rx", "old_ry", "old_rz", "mx", "my", "mz", "rx", "ry", "rz", "sqm", "L_01", "L_02", "L_03", "L_04", "L_05", "L_12", "L_13", "L_14", "L_15", "L_23", "L_24", "L_25", "L_34", "L_35", "L_45", "E",  "m.r", "r*r"])#return results of simulation
        elif args.model =="P3D" or args.model == "K3D":
            data = np.hstack((ts, ms_old, rs_old, ms, rs, msqs, Ls[:,0,1].reshape(-1,1), Ls[:,0,2].reshape(-1,1), Ls[:,0,3].reshape(-1,1), Ls[:,0,4].reshape(-1,1), Ls[:,0,5].reshape(-1,1), Ls[:,1,2].reshape(-1,1), Ls[:,1,3].reshape(-1,1), Ls[:,1,4].reshape(-1,1), Ls[:,1,5].reshape(-1,1), Ls[:,2,3].reshape(-1,1), Ls[:,2,4].reshape(-1,1), Ls[:,2,5].reshape(-1,1), Ls[:,3,4].reshape(-1,1), Ls[:,3,5].reshape(-1,1), Ls[:,4,5].reshape(-1,1), Es))
            return pd.DataFrame(data, columns = ["time", "old_mx", "old_my", "old_mz", "old_rx", "old_ry", "old_rz", "mx", "my", "mz", "rx", "ry", "rz", "sqm", "L_01", "L_02", "L_03", "L_04", "L_05", "L_12", "L_13", "L_14", "L_15", "L_23", "L_24", "L_25", "L_34", "L_35", "L_45", "E"])#return results of simulation
        elif args.model =="P2D":
            data = np.hstack((ts, ms_old, rs_old, ms, rs, msqs, Ls[:,0,1].reshape(-1,1), Ls[:,0,2].reshape(-1,1), Ls[:,0,3].reshape(-1,1), Ls[:,1,2].reshape(-1,1), Ls[:,1,3].reshape(-1,1), Ls[:,2,3].reshape(-1,1), Es))
            return pd.DataFrame(data, columns = ["time", "old_mx", "old_my", "old_rx", "old_ry", "mx", "my",  "rx", "ry", "sqm", "L_01", "L_02", "L_03", "L_12", "L_13", "L_23", "E"])#return results of simulation
        elif args.model =="Sh":
            data = np.hstack((ts, ms_old, ms, Ls[:,0,1].reshape(-1,1), Ls[:,0,2].reshape(-1,1), Ls[:,0,3].reshape(-1,1), Ls[:,1,2].reshape(-1,1), Ls[:,1,3].reshape(-1,1), Ls[:,2,3].reshape(-1,1), Es))
            return pd.DataFrame(data, columns = ["time", "old_u", "old_x", "old_y", "old_z", "u", "x",  "y", "z", "L_01", "L_02", "L_03", "L_12", "L_13", "L_23", "E"])#return results of simulation


def simulate_batch(args, initial_states_batch, method = "normal"):
    #Create d2E matrix
    d2E = torch.tensor([[1/args.Ix,0,0],\
                    [0,1/args.Iy,0],\
                    [0,0, 1/args.Iz]], device=args.device)
                    
    ts, ms, msqs, Ls, Es, casss, rs, mrs, rsqs = [], [], [], [], [], [], [], [], []

    # device = torch.device("cuda" if torch.cuda.is_available() and args.cuda and args.simulate_cuda else "cpu")
    device = args.device
    batch_size = initial_states_batch.shape[0]

    init_mx = initial_states_batch[:, 0]
    init_my = initial_states_batch[:, 1]
    init_mz = initial_states_batch[:, 2]
    init_rx = initial_states_batch[:, 3]
    init_ry = initial_states_batch[:, 4]
    init_rz = initial_states_batch[:, 5]

    if args.model == "RB": # Rigid body
        if method == "implicit":
            if args.scheme == "IMR":
                solver = RBNeuralIMR(args.Ix,args.Iy,args.Iz, d2E, init_mx,init_my,init_mz,args.dt,args.alpha, method = "implicit", name = args.folder_name, device=device)
            else:
                solver = Neural(args.Ix,args.Iy,args.Iz, d2E, init_mx,init_my,init_mz,args.dt,args.alpha, method = "implicit", name = args.folder_name, device=device)
        elif method == "soft":
            if args.scheme == "IMR":
                solver = RBNeuralIMR(args.Ix,args.Iy,args.Iz, d2E, init_mx,init_my,init_mz,args.dt,args.alpha, method = "soft", name = args.folder_name, device=device)
            else:
                solver = Neural(args.Ix,args.Iy,args.Iz, d2E, init_mx,init_my,init_mz,args.dt,args.alpha, method = "soft", name = args.folder_name, device=device)
        elif method == "without":
            if args.scheme == "IMR":
                solver = RBNeuralIMR(args.Ix,args.Iy,args.Iz, d2E, init_mx,init_my,init_mz,args.dt,args.alpha, method = "without", name = args.folder_name, device=device)
            else:
                solver = Neural(args.Ix,args.Iy,args.Iz, d2E, init_mx,init_my,init_mz,args.dt,args.alpha, method = "without", name = args.folder_name, device=device)
        elif method =="normal":
            if args.scheme == "Eh":
                solver = RBEhrenfest(args.Ix,args.Iy,args.Iz, d2E, init_mx,init_my,init_mz,args.dt,args.alpha,device=device)
            elif args.scheme == "CN":
                solver = RBESeReCN(args.Ix,args.Iy,args.Iz, d2E, init_mx,init_my,init_mz,args.dt,args.alpha,device=device)
            elif args.scheme == "FE":
                solver = RBESeReFE(args.Ix,args.Iy,args.Iz, d2E, init_mx,init_my,init_mz,args.dt,args.alpha,device=device)
            elif args.scheme == "IMR":
                solver = RBIMR(args.Ix,args.Iy,args.Iz, d2E, init_mx,init_my,init_mz,args.dt,device=device)
            elif args.scheme == "RK4": 
                solver = RBRK4(args.Ix,args.Iy,args.Iz, d2E, init_mx,init_my,init_mz,args.dt,args.M_tau,device=device)
            else:
                raise Exception("Unknown scheme.")
        else:
            raise Exception("Unknown method.")

    elif args.model == "HT": # Heavy top
        """if method == "implicit":
            raise Exception("Implicit solver not yet implemented for HT.")
            #solver = NeuralHT(args.Ix,args.Iy,args.Iz, d2E, init_mx,init_my,init_mz,args.dt,args.alpha, method = "implicit")"""
        if method == "soft":
            if args.scheme == "IMR":
                solver = HeavyTopNeuralIMR(args.Ix,args.Iy,args.Iz, d2E, init_mx,init_my,init_mz,args.dt,args.alpha, args.Mgl, init_rx, init_ry, init_rz, device=device, method = "soft", name = args.folder_name)
            else:
                solver = HeavyTopNeural(args.Ix,args.Iy,args.Iz, d2E, init_mx,init_my,init_mz,args.dt,args.alpha, args.Mgl, init_rx, init_ry, init_rz, device=device, method = "soft", name = args.folder_name)
        elif method == "without":
            if args.scheme == "IMR":
                solver = HeavyTopNeuralIMR(args.Ix,args.Iy,args.Iz, d2E, init_mx,init_my,init_mz,args.dt,args.alpha, args.Mgl, init_rx, init_ry, init_rz, device=device, method = "without", name = args.folder_name)
            else:
                solver = HeavyTopNeural(args.Ix,args.Iy,args.Iz, d2E, init_mx,init_my,init_mz,args.dt,args.alpha, args.Mgl, init_rx, init_ry, init_rz, device=device, method = "without", name = args.folder_name)
        elif method =="normal":
            #if args.scheme == "Eh":
            #    solver = RBEhrenfest(args.Ix,args.Iy,args.Iz, d2E, init_mx,init_my,init_mz,args.dt,args.alpha)
            #elif args.scheme == "CN":
            if args.scheme == "IMR":
                solver = HeavyTopIMR(args.Ix,args.Iy,args.Iz, d2E, init_mx,init_my,init_mz,args.dt,args.alpha, args.Mgl, init_rx, init_ry, init_rz, device=device)
            else:
                solver = HeavyTopCN(args.Ix,args.Iy,args.Iz, d2E, init_mx,init_my,init_mz,args.dt,args.alpha, args.Mgl, init_rx, init_ry, init_rz, device=device)
            #elif args.scheme == "FE":
            #    solver = RBESeReFE(args.Ix,args.Iy,args.Iz, d2E, init_mx,init_my,init_mz,args.dt,args.alpha)
            #else:
            #    raise Exception("Unknown scheme.")
        else:
            raise Exception("Unknown method.")
    elif args.model == "P3D" or args.model == "K3D": # Particle in 3D or Kepler problem in 3D (not tested yet)
        """if method == "implicit":
            raise Exception("Implicit solver not yet implemented for P3D.")
            #solver = NeuralHT(args.Ix,args.Iy,args.Iz, d2E, init_mx,init_my,init_mz,args.dt,args.alpha, method = "implicit")"""
        if method == "soft":
            if args.scheme == "IMR":
                solver = Particle3DNeuralIMR(args.M, args.dt, args.alpha, init_rx, init_ry, init_rz, init_mx,init_my,init_mz,device=device, method = "soft", name = args.folder_name)
            else:
                solver = Particle3DNeural(args.M, args.dt, args.alpha, init_rx, init_ry, init_rz, init_mx,init_my,init_mz,device=device, method = "soft", name = args.folder_name)
        elif method == "without":
            if args.scheme == "IMR":
                solver = Particle3DNeuralIMR(args.M, args.dt, args.alpha, init_rx, init_ry, init_rz, init_mx,init_my,init_mz,device=device, method = "without", name = args.folder_name)
            else:
                solver = Particle3DNeural(args.M, args.dt, args.alpha, init_rx, init_ry, init_rz, init_mx,init_my,init_mz,device=device, method = "without", name = args.folder_name)
        elif method =="normal":
            #if args.scheme == "Eh":
            #    solver = RBEhrenfest(args.Ix,args.Iy,args.Iz, d2E, init_mx,init_my,init_mz,args.dt,args.alpha)
            #elif args.scheme == "CN":
            if args.scheme == "IMR":
                if args.model == "K3D":
                    solver = Particle3DKeplerIMR(args.M, args.dt, args.alpha, init_rx, init_ry, init_rz, init_mx,init_my,init_mz, device=device)
                else:
                    solver = Particle3DIMR(args.M, args.dt, args.alpha, init_rx, init_ry, init_rz, init_mx,init_my,init_mz, device=device)
            else:
                """if args.model == "K3D":
                    raise Exception("Don't use CN for Kepler.")"""
                solver = Particle3DCN(args.M, args.dt, args.alpha, init_rx, init_ry, init_rz, init_mx,init_my,init_mz, device=device)
            #elif args.scheme == "FE":
            #    solver = RBESeReFE(args.Ix,args.Iy,args.Iz, d2E, init_mx,init_my,init_mz,args.dt,args.alpha)
            #else:
            #    raise Exception("Unknown scheme.")
    elif args.model == "P2D": # Particle in 2D
        """if method == "implicit":
            raise Exception("Implicit solver not yet implemented for P2D.")
            #solver = NeuralHT(args.Ix,args.Iy,args.Iz, d2E, init_mx,init_my,init_mz,args.dt,args.alpha, method = "implicit")"""
        if method == "soft":
            if args.scheme == "IMR":
                solver = Particle2DNeural(args.M, args.dt, args.alpha, init_rx, init_ry, init_mx,init_my, zeta = args.zeta, device=device, method = "soft", name = args.folder_name)
            """else:
                raise Exception("Not implemented.")"""
                #solver = Particle3DNeural(args.M, args.dt, args.alpha, init_rx, init_ry, init_rz, init_mx,init_my,init_mz, method = "soft", name = args.folder_name)
        elif method == "without":
            if args.scheme == "IMR":
                solver = Particle2DNeural(args.M, args.dt, args.alpha, init_rx, init_ry, init_mx,init_my, zeta = args.zeta, device=device, method = "without", name = args.folder_name)
            """else:
                raise Exception("Not implemented.")"""
                #solver = Particle3DNeural(args.M, args.dt, args.alpha, init_rx, init_ry, init_rz, init_mx,init_my,init_mz, method = "without", name = args.folder_name)
        elif method =="normal":
            #if args.scheme == "Eh":
            #    solver = RBEhrenfest(args.Ix,args.Iy,args.Iz, d2E, init_mx,init_my,init_mz,args.dt,args.alpha)
            #elif args.scheme == "CN":
            if args.scheme == "IMR":
                solver = Particle2DIMR(args.M, args.dt, args.alpha, init_rx, init_ry, init_mx,init_my,args.zeta, device=device)
            """else:
                raise Exception("Not implemented.")"""
                #solver = Particle3DCN(args.M, args.dt, args.alpha, init_rx, init_ry, init_rz, init_mx,init_my,init_mz)
        else:
            raise Exception("Unknown method.")
    elif args.model == "Sh": # Shivamoggi equations
        mask = init_ry.abs() > 0.1
        init_ry[mask] = init_ry[mask] / (10 * init_ry[mask].abs())
        """if method == "implicit":
            raise Exception("Implicit solver not yet implemented for Shivamoggi.")
            #solver = NeuralHT(args.Ix,args.Iy,args.Iz, d2E, init_mx,init_my,init_mz,args.dt,args.alpha, method = "implicit")"""
        if method == "soft":
            if args.scheme == "IMR":
                solver = ShivamoggiNeural(args.M, args.dt, args.alpha, init_rx, init_ry, init_rz,init_mx, device=device, method = "soft", name = args.folder_name)
            """else:
                raise Exception("Not implemented.")"""
                #solver = Particle3DNeural(args.M, args.dt, args.alpha, init_rx, init_ry, init_rz, init_mx,init_my,init_mz, method = "soft", name = args.folder_name)
        elif method == "without":
            if args.scheme == "IMR":
                solver = ShivamoggiNeural(args.M, args.dt, args.alpha, init_rx, init_ry, init_rz,init_mx, device=device, method = "without", name = args.folder_name)
            """else:
                raise Exception("Not implemented.")"""
                #solver = Particle3DNeural(args.M, args.dt, args.alpha, init_rx, init_ry, init_rz, init_mx,init_my,init_mz, method = "without", name = args.folder_name)
        elif method =="normal":
            #if args.scheme == "Eh":
            #    solver = RBEhrenfest(args.Ix,args.Iy,args.Iz, d2E, init_mx,init_my,init_mz,args.dt,args.alpha)
            #elif args.scheme == "CN":
            if args.scheme == "IMR":
                solver = ShivamoggiIMR(args.M, args.dt, args.alpha, init_rx, init_ry, init_rz,init_mx, device=device)
            """else:
                raise Exception("Not implemented.")"""
                #solver = Particle3DCN(args.M, args.dt, args.alpha, init_rx, init_ry, init_rz, init_mx,init_my,init_mz)
        else:
            raise Exception("Unknown method.")

    #Timesteps
    dt = args.dt

    #Preparing files for output
    if args.model == "RB":
        m = torch.stack([init_mx,init_my,init_mz], dim=1) # set dim=1
        r = torch.stack([init_rx,init_ry,init_rz], dim=1) #not used. Just not to cause errors
        Ls.append(solver.get_L(m))
        Es.append(solver.get_E(m))
    elif args.model == "HT":
        m = torch.stack([init_mx,init_my,init_mz], dim=1)
        r = torch.stack([init_rx,init_ry,init_rz], dim=1)
        mr = torch.cat([m,r], dim=1)
        Ls.append(solver.get_L(mr))
        Es.append(solver.get_E(mr))
    elif args.model == "P3D" or args.model == "K3D":
        m = torch.stack([init_mx,init_my,init_mz], dim=1)
        r = torch.stack([init_rx,init_ry,init_rz], dim=1)
        mr = torch.cat([r,m], dim=1)
        Ls.append(solver.get_L(mr))
        Es.append(solver.get_E(mr))
    elif args.model == "P2D":
        m = torch.stack([init_mx,init_my], dim=1)
        r = torch.stack([init_rx,init_ry], dim=1)
        mr = torch.cat([r,m], dim=1)
        Ls.append(solver.get_L(mr))
        Es.append(solver.get_E(mr))
    elif args.model == "Sh":
        r = torch.stack([init_mx,init_rx,init_ry,init_rz], dim=1) #not used
        m = torch.stack([init_mx,init_rx,init_ry,init_rz], dim=1)
        Ls.append(solver.get_L(m))
        Es.append(solver.get_E(m))
    else:
        raise Exception("Unknown model.")

    msqs.append((m * m).sum(dim=1))
    ts.append(0)
    ms.append(m)
    rs.append(r)
    mrs.append((m * r).sum(dim=1))
    rsqs.append((r * r).sum(dim=1))
    if method == "implicit":
        casss.append(solver.get_cass(m)) #not yet implemented for HT
    else:
        casss.append(0.0) #not implemented yet

    store_each = 1

    #calculate evolution
    for i in range(args.steps-1):
        #print(i)
        # m_old = m.clone()
        # r_old = r.clone()

        if args.model == "RB" or args.model == "Sh":
            m = solver.m_new()
            #print(torch.max(m))
        elif args.model == "HT":
            (m, r) = solver.m_new()
        elif args.model == "P3D" or args.model == "K3D" or args.model=="P2D":
            (r, m) = solver.m_new()

        if i % store_each == 0:
            t = torch.full((batch_size, 1), dt * i, device=m.device, dtype=torch.float64)
            ts.append(t)

            ms.append(m)
            msqs.append((m * m).sum(dim=1, keepdim=True))
            if args.model == "RB" or args.model == "Sh":
                Ls.append(solver.get_L(m))
                Es.append(solver.get_E(m))
            elif args.model == "HT":
                mr = torch.cat([m,r], dim=1)
                Ls.append(solver.get_L(mr))
                Es.append(solver.get_E(mr))
                rs.append(r)
                mrs.append((m * r).sum(dim=1, keepdim=True))
                rsqs.append((r * r).sum(dim=1, keepdim=True))
            elif args.model == "P3D" or args.model == "K3D":
                Ls.append(solver.get_L(mr))
                Es.append(solver.get_E(mr))
                rs.append(r)
                rsqs.append((r * r).sum(dim=1))
            elif args.model == "P2D":
                Ls.append(solver.get_L(mr))
                Es.append(solver.get_E(mr))
                rs.append(r)
                rsqs.append((r * r).sum(dim=1, keepdim=True))
            if method == "implicit":
                casss.append(solver.get_cass(m))
            else:
                casss.append(torch.zeros(batch_size, 1, device=device)) #Casimirs implemented only in the implicit case, so far

        if args.verbose and (i % (max(1,int(round(args.steps/100)))) == 0):
            m_norm0 = torch.norm(m[0])
            E0 = 0.5 * m[0] @ d2E @ m[0]

            print(f"{i/args.steps*100:.1f}%")
            print(f"|m| (sample 0) = {m_norm0.item():.4f}")
            print(f"E   (sample 0) = {E0.item():.6f}")

    ts = torch.cat(ts[1:], dim=0).view(-1, batch_size, 1).cpu().numpy()

    ms_old = torch.stack(ms[:-1]).transpose(0, 1).detach().cpu().numpy()
    ms = torch.stack(ms[1:]).transpose(0, 1).detach().cpu().numpy()
    msqs = torch.cat(msqs[1:], dim=0).view(-1, batch_size, 1).detach().cpu().numpy()
    Es = torch.cat(Es[1:], dim=0).view(-1, batch_size, 1).detach().cpu().numpy()
    casss = torch.cat(casss[1:], dim=0).view(-1, batch_size, 1).detach().cpu().numpy()

    if rs and len(rs) > 1:
        rs_old = torch.stack(rs[:-1]).transpose(0, 1).detach().cpu().numpy()
        rs = torch.stack(rs[1:]).transpose(0, 1).detach().cpu().numpy()
        if args.model == "HT":
            mrs = torch.cat(mrs[1:], dim=0).view(-1, batch_size, 1).detach().cpu().numpy()
            rsqs = torch.cat(rsqs[1:], dim=0).view(-1, batch_size, 1).detach().cpu().numpy()
        else:
            mrs = rsqs = None
        rs_old = rs_old.transpose(1, 0, 2)
        rs = rs.transpose(1, 0, 2)
    else:
        rs_old = rs = mrs = rsqs = None

    Ls = torch.stack(Ls[1:], dim=0)
    i_indices, j_indices = torch.triu_indices(Ls.shape[-2], Ls.shape[-1], offset=1)
    Ls_upper = Ls[:, :, i_indices, j_indices]
    Ls_upper_np = Ls_upper.detach().cpu().numpy()

    L_columns = [f"L_{i}{j}" for i, j in zip(i_indices.tolist(), j_indices.tolist())]

    Ls_upper_np = Ls_upper_np.reshape(args.steps-1, batch_size, -1)
    ms_old = ms_old.transpose(1, 0, 2)
    ms = ms.transpose(1, 0, 2)

    # print(ts.shape, ms_old.shape, ms.shape, msqs.shape, Ls_upper_np.shape, Es.shape, casss.shape)

    if method == "implicit": #returns also the Casimir
        data = np.concatenate((ts, ms_old, ms, msqs, Ls_upper_np, Es, casss), axis=-1)
        T, B, F = data.shape
        data_flat = data.transpose(1, 0, 2).reshape(B * T, F)
        return pd.DataFrame(data_flat, columns = ["time", "old_mx", "old_my", "old_mz", "mx", "my", "mz", "sqm"] + L_columns + ["E", "cass"])#return results of simulation
    else:
        if args.model == "RB":
            data = np.concatenate((ts, ms_old, ms, msqs, Ls_upper_np, Es), axis=-1)
            T, B, F = data.shape
            data_flat = data.transpose(1, 0, 2).reshape(B * T, F)
            return pd.DataFrame(data_flat, columns = ["time", "old_mx", "old_my", "old_mz", "mx", "my", "mz", "sqm"] + L_columns + ["E"])#return results of simulation
        elif args.model =="HT":
            data = np.concatenate((ts, ms_old, rs_old, ms, rs, msqs, Ls_upper_np, Es, mrs, rsqs), axis=-1)
            T, B, F = data.shape
            data_flat = data.transpose(1, 0, 2).reshape(B * T, F)
            return pd.DataFrame(data_flat, columns = ["time", "old_mx", "old_my", "old_mz", "old_rx", "old_ry", "old_rz", "mx", "my", "mz", "rx", "ry", "rz", "sqm"] + L_columns + ["E",  "m.r", "r*r"])#return results of simulation
        elif args.model =="P3D" or args.model == "K3D":
            data = np.concatenate((ts, ms_old, rs_old, ms, rs, msqs, Ls_upper_np, Es), axis=-1)
            T, B, F = data.shape
            data_flat = data.transpose(1, 0, 2).reshape(B * T, F)
            return pd.DataFrame(data_flat, columns = ["time", "old_mx", "old_my", "old_mz", "old_rx", "old_ry", "old_rz", "mx", "my", "mz", "rx", "ry", "rz", "sqm"] + L_columns + ["E"])#return results of simulation
        elif args.model =="P2D":
            data = np.concatenate((ts, ms_old, rs_old, ms, rs, msqs, Ls_upper_np, Es), axis=-1)
            T, B, F = data.shape
            data_flat = data.transpose(1, 0, 2).reshape(B * T, F)
            return pd.DataFrame(data_flat, columns = ["time", "old_mx", "old_my", "old_rx", "old_ry", "mx", "my",  "rx", "ry", "sqm"] + L_columns + ["E"])#return results of simulation
        elif args.model =="Sh":
            data = np.concatenate((ts, ms_old, ms, Ls_upper_np, Es), axis=-1)
            T, B, F = data.shape
            data_flat = data.transpose(1, 0, 2).reshape(B * T, F)
            return pd.DataFrame(data_flat, columns = ["time", "old_u", "old_x", "old_y", "old_z", "u", "x",  "y", "z"] + L_columns + ["E"])#return results of simulation


#def get_file_name(args):
#    if args.generate:
#        file_name = args.folder_name+"/data/dataset.xyz"
#    elif args.implicit:
#        file_name = args.folder_name+"/data/learned_implicit.xyz"
#    elif args.soft:
#        file_name = args.folder_name+"/data/learned_soft.xyz"
#    elif args.without:
#        file_name = args.folder_name+"/data/learned_without.xyz"
#    else:
#        file_name = args.folder_name+"/data/m.xyz"
#    return file_name



# The above code is a Python script that defines a command-line interface for a simulation program. It
# uses the argparse module to parse command-line arguments and configure the simulation parameters.
# The script then checks which methods are specified and calls the simulate function with the
# corresponding method argument. Finally, it saves the simulation data to a file based on the
# specified method.
if __name__ == "__main__":
    #Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", default="CN", type=str, help="Numerical scheme. Eh Ehrenfest. FE forward euler, CN Crank-Nicholson")
    parser.add_argument("--steps", default=500, type=int, help="Number of simulation steps")
    parser.add_argument("--init_mx", default=1.0, type=float, help="Initial momentum, x component")
    parser.add_argument("--init_my", default=0.3, type=float, help="Initial momentum, y component")
    parser.add_argument("--init_mz", default=0.3, type=float, help="Initial momentum, z component")
    parser.add_argument("--Ix", default=10.0, type=float, help="Ix")
    parser.add_argument("--Iy", default=20.0, type=float, help="Iy")
    parser.add_argument("--Iz", default=40.0, type=float, help="Iz")
    parser.add_argument("--dt", default=0.08, type=float, help="Timestep")
    parser.add_argument("--alpha", default=0.01, type=float, help="A coeficient")
    parser.add_argument("--normalise", default=False, action="store_true", help="Normalise energy matrix at the end")
    parser.add_argument("--generate", default=False, action="store_true", help="Save trajectory for learning")
    parser.add_argument("--verbose", default=False, action="store_true", help="Print a lot of useful output")
    parser.add_argument("--implicit", default=False, action="store_true", help="Use implicit Jacobi neural solver.")
    parser.add_argument("--soft", default=False, action="store_true", help="Use penalty in loss function to learn Jacobi identity. Otherwise it is implicitly valid")
    parser.add_argument("--without", default=False, action="store_true", help="Trained without Jacobi.")
    parser.add_argument("--plot", default=False, action="store_true", help="Plot results")
    parser.add_argument("--model", default="RB", type=str, help="Model: RB, HT, P3D, or K3D")
    parser.add_argument("--Mgl", default=9.81*sqrt(3)/2, type=float, help="M*g*l")
    parser.add_argument("--init_rx", default=0.0, type=float, help="Initial r, x component")
    parser.add_argument("--init_ry", default=-0.195090, type=float, help="Initial r, y component")
    parser.add_argument("--init_rz", default=0.980785, type=float, help="Initial r, z component")
    parser.add_argument("--M", default=10.5, type=float, help="mass")
    parser.add_argument("--cuda", default=False, action="store_true", help="Use GPU if available")
    

    args = parser.parse_args([] if "__file__" not in globals() else None)

    if args.cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        args.device = torch.device("cpu")
        print("Using CPU")

    #checking whether more methods are specified (error), only one is possible
    methods = 0
    if args.generate:
        methods += 1
    if args.implicit:
        methods += 1
    if args.without:
        methods += 1
    if args.soft:
        methods += 1
    if methods > 1:
        raise Exception("Too many methods specified.")

    if args.generate:
        data = simulate(args, method = "normal")
        save_simulation(data, args.folder_name+"/data/dataset.xyz")
    if args.implicit:
        data = simulate(args, method = "implicit")
        save_simulation(data, args.folder_name+"/data/learned_implicit.xyz")
    if args.without:
        data = simulate(args, method = "without")
        save_simulation(data, args.folder_name+"/data/learned_without.xyz")
    if args.soft:
        data = simulate(args, method = "soft")
        save_simulation(data, args.folder_name+"/data/learned_soft.xyz")

    #if args.plot:
    #    data_frame = pd.read_csv(file_name)
    #    plt.plot(data_frame["time"], data_frame["sqm"], lw=0.7, label="msq")
    #    plt.legend()
    #    plt.show()
    #    plt.plot(data_frame["time"], data_frame["E"], lw=0.7, label="E")
    #    plt.legend()
    #    plt.show()
    #    plt.plot(data_frame["time"], data_frame["mx"], lw=0.7, label="mx")
    #    plt.legend()
    #    plt.show()
    #    plt.plot(data_frame["time"], data_frame["my"], lw=0.7, label="my")
    #    plt.legend()
    #    plt.show()
    #    plt.plot(data_frame["time"], data_frame["mz"], lw=0.7, label="mz")
    #    plt.legend()
    #    plt.show()
