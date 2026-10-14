"""Deterministic 3D structured binary test volumes."""
import numpy as np

def volume(n, pattern='mixed'):
    x,y,z=np.ogrid[:n,:n,:n]
    fg=np.zeros((n,n,n),dtype=bool)
    if pattern=='mixed':
        cx,cy,cz=(n*.31,n*.31,n*.34)
        rho=np.sqrt((x-cx)**2+(y-cy)**2)
        torus=(rho-n*.13)**2+(z-cz)**2 <= max(2.,n*.042)**2
        sx,sy,sz=n*.70,n*.70,n*.67
        rr=np.sqrt((x-sx)**2+(y-sy)**2+(z-sz)**2)
        shell=(rr>=n*.108)&(rr<=n*.108+max(2.0,n*.035))
        fg=torus|shell
    elif pattern=='lattice':
        block=n/3
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    cx,cy,cz=(i+.5)*block,(j+.5)*block,(k+.5)*block
                    if (i+j+k)%2:
                        rho=np.sqrt((x-cx)**2+(y-cy)**2)
                        fg|=(rho-block*.20)**2+(z-cz)**2<=max(1.25,block*.085)**2
                    else:
                        rr=np.sqrt((x-cx)**2+(y-cy)**2+(z-cz)**2)
                        fg|=(rr>=block*.20)&(rr<=block*.20+max(1.35,block*.065))
    elif pattern=='pants_cavity':
        # Spatially separated multilevel H1 pair-of-pants and an H2 shell.
        pants=np.array([
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,30,30,30,30,30,30,30,30,30,0],
            [0,30,10,10,10,30,20,20,20,30,0],
            [0,30,10,90,10,30,20,80,20,30,0],
            [0,30,10,10,10,30,20,20,20,30,0],
            [0,30,30,30,30,30,30,30,30,30,0],
            [0,0,0,0,0,0,0,0,0,0,0]],dtype=np.uint8)
        factor=max(1,n//32)
        a=np.repeat(np.repeat(pants,factor,axis=0),factor,axis=1)
        z0,y0,x0=int(n*.20),int(n*.09),int(n*.09)
        t=max(2,factor)
        result=np.full((n,n,n),200,dtype=np.uint8)
        result[z0:z0+t,y0:y0+a.shape[0],x0:x0+a.shape[1]]=a
        sx,sy,sz=n*.73,n*.73,n*.73
        rr=np.sqrt((x-sx)**2+(y-sy)**2+(z-sz)**2)
        shell=(rr>=n*.105)&(rr<=n*.105+max(2.0,n*.034))
        result[shell]=0
        return result
    else:
        raise ValueError(pattern)
    return np.where(fg,0,200).astype(np.uint8)
