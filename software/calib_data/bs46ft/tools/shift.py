import sys, cv2, numpy as np
a=cv2.imread(sys.argv[1],0).astype(np.float32); b=cv2.imread(sys.argv[2],0).astype(np.float32)
for name,(x0,y0,x1,y1) in {"rack":(660,140,800,290),"doorframe":(420,40,520,460),"blinds":(470,40,660,200)}.items():
    t=a[y0:y1,x0:x1]; s=b[y0-40:y1+40,x0-40:x1+40]
    r=cv2.matchTemplate(s,t,cv2.TM_CCOEFF_NORMED); _,mx,_,(px,py)=cv2.minMaxLoc(r)
    # subpixel parabola
    def sp(v,i):
        return 0 if i<=0 or i>=len(v)-1 else 0.5*(v[i-1]-v[i+1])/(v[i-1]-2*v[i]+v[i+1])
    dx=px-40+sp(r[py,:],px); dy=py-40+sp(r[:,px],py)
    print(f"{name:9s} dx {dx:+.2f} dy {dy:+.2f} (corr {mx:.3f})")
