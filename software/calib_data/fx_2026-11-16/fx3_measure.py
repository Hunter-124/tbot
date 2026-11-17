import cv2, numpy as np, json
FT=0.3048; S=0.515
# group -> (snaps, ft, row band, rough left, rough right)  rough edges read off the ruled views
G=[([36,37],10,(560,640),262,938),([34,35],15,(520,600),340,780),([32,33],20,(500,560),352,690),
   ([30,31],25,(470,530),402,660),([28,29],30,(440,500),430,640)]
def cross(p, x_from, x_to, level):
    """sub-pixel x where profile p crosses `level` scanning x_from -> x_to (indices)."""
    step = 1 if x_to > x_from else -1
    for x in range(x_from, x_to, step):
        a, b = p[x], p[x + step]
        if (a - level) * (b - level) <= 0 and a != b:
            return x + step * (level - a) / (b - a)
    return None
rows=[]
for snaps, ft, band, xl, xr in G:
    for i in snaps:
        im=cv2.imread(f"snaps/snap_{i:04d}.png",0).astype(np.float64)
        p=cv2.GaussianBlur(im[band[0]:band[1]].mean(axis=0).reshape(1,-1),(0,0),2).ravel()
        mid=(xl+xr)//2; inside=np.median(p[xl+70:xr-70]) if xr-xl>200 else np.median(p[mid-15:mid+15])
        outL=np.median(p[max(0,xl-130):xl-70]); outR=np.median(p[xr+70:min(len(p),xr+130)])
        L=cross(p, xl+60, xl-60, 0.5*(inside+outL)); R=cross(p, xr-60, xr+60, 0.5*(inside+outR))
        w=None if L is None or R is None else R-L
        rows.append(dict(snap=i,ft=ft,L=L,R=R,w=w,inside=inside,outL=outL,outR=outR))
        if w is None:
            print(f"snap {i} {ft:>2} ft: no clean edge crossing (background too close to the box level)"); continue
        print(f"snap {i} {ft:>2} ft: L {L:7.1f} R {R:7.1f} width {w:6.1f} px  levels in {inside:5.1f} outL {outL:5.1f} outR {outR:5.1f}  f={w*ft*FT/S:6.0f}")
json.dump(rows,open("fx3_rows.json","w"),indent=1)
