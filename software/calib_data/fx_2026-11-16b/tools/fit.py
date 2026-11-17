# usage: python fit.py [FT=height/width ...]  (focal length per distance and 1/size-vs-distance fit; built-in table = this session)
import numpy as np, sys
S=0.2159; SW=0.2794
data={45:(63.77,81.86),40:(71.44,91.81),35:(81.33,104.47),30:(94.87,122.14)}
data.update({int(k):tuple(map(float,v.split('/'))) for k,v in (a.split('=') for a in sys.argv[1:])})
ft=np.array(sorted(data,reverse=True)); d=ft*0.3048
for name,col,size in (("height",0,S),("width",1,SW)):
    h=np.array([data[k][col] for k in ft])
    f0=h*d/size
    A=np.c_[d,np.ones(len(d))]; (a,b),*_=np.linalg.lstsq(A,1/h,rcond=None); f=1/(a*size); z0=b/a
    r=(1/h-(a*d+b))*h*100
    # z0=0 fit: f = mean of f0 weighted; residual
    print(f"{name}: per-distance f(z0=0) {dict(zip(ft.tolist(),f0.round(1)))}  mean {f0.mean():.1f} sd {f0.std(ddof=1):.1f}")
    # bootstrap free fit
    if len(d)>2:
        print(f"   free z0: f {f:.1f} z0 {z0*100:+.1f} cm  resid % {r.round(3)}")
    # implied distance error at each if f=4000/z0=0
