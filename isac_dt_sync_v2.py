"""
isac_dt_sync_v2.py  — Option B: Two-scenario integrity-aware DT sync.

Scenario 1: Good upstream filter  (adaptive-R + NIS gate, Paper 2)
Scenario 2: Degraded upstream     (fixed-R KF, no gate)

Four sync policies compared in each scenario:
  unconditional  — sync every step  (800/trial)
  periodic       — every T_period steps
  event          — when D_k > tau_D  (quality-blind)
  proposed       — SYNC/FUSE/HOLD on D_risk = D_k / (C_k + delta)

Primary result: Pareto curve (twin RMSE vs sync count) for each scenario.
Secondary: bar chart of twin RMSE and mean AoI at default thresholds.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.linalg import solve_discrete_are
from scipy.stats import chi2

# ─── Constants ────────────────────────────────────────────────────
DT          = 0.05
N_STEPS     = 800
N_SEEDS     = 30
SIGMA_ACCEL = 0.55
BASE_SIGMA  = 0.065
Q_MIN       = 0.04
ALPHA_CONF  = 1.0
DELTA_CONF  = 0.1
TAU_LOW     = 0.25
TAU_HIGH    = 1.50
FUSE_W      = 0.50
T_PERIOD    = 10
TAU_EVENT   = 0.30
MAX_HOLD    = 20          # anti-spiral: force FUSE after this many HOLDs
NIS_SOFT    = float(chi2.ppf(0.95,  df=3))   # 7.815
NIS_HARD    = float(chi2.ppf(1-1e-6,df=3))   # 30.67

# ─── Motion / measurement ─────────────────────────────────────────
def _matrices(dt):
    e3=np.eye(3); z3=np.zeros((3,3))
    F  = np.block([[e3,dt*e3],[z3,e3]])
    H  = np.block([e3,z3])
    sa = SIGMA_ACCEL
    qp=0.25*dt**4*sa**2; qc=0.5*dt**3*sa**2; qv=dt**2*sa**2
    Q  = np.block([[qp*e3,qc*e3],[qc*e3,qv*e3]])
    R0 = np.eye(3)*BASE_SIGMA**2
    return F,H,Q,R0

def make_trajectory(n,dt,rng):
    t  = np.linspace(0,40,n)
    px = 3.0*np.sin(2*np.pi*t/40)
    py = 1.5*np.sin(4*np.pi*t/40)
    pz = 1.5+0.3*np.sin(2*np.pi*t/20)
    vx,vy,vz = (np.gradient(c,dt) for c in [px,py,pz])
    return np.c_[px,py,pz], np.c_[vx,vy,vz]

def make_quality(n,rng):
    tf = np.linspace(0,1,n)
    q  = 0.85-0.20*rng.random(n)
    sig= lambda x,c,w: 1/(1+np.exp(-(x-c)/w))
    q -= sig(tf,.25,.01)*sig(-tf,-.35,.01)*(1-0.22)*q
    q -= sig(tf,.65,.01)*sig(-tf,-.75,.01)*(1-0.35)*q
    return np.clip(q,Q_MIN,1.0)

def make_measurements(pos,q,rng):
    sigma = BASE_SIGMA/np.sqrt(np.clip(q,Q_MIN,1.0))
    meas  = pos+rng.normal(0,sigma[:,None],(len(q),3))
    rand_out = rng.random(len(q)) < (0.01+0.08*(1-q))
    burst = ((np.arange(len(q))%31)<4) & \
            (np.arange(len(q))>len(q)//4) & \
            (np.arange(len(q))<3*len(q)//4)
    mask = rand_out|burst
    dirs = rng.normal(0,1,(len(q),3))
    dirs/= np.linalg.norm(dirs,axis=1,keepdims=True)+1e-9
    mags = rng.uniform(.8,2.2,len(q))*(1+1.8*(1-q))
    meas[mask] += dirs[mask]*mags[mask,None]
    return meas

# ─── Two upstream filter modes ────────────────────────────────────
class UpstreamFilter:
    """
    mode='good'  : adaptive-R + NIS gate (Paper 2 proposed)
    mode='fixed' : fixed-R KF, no gate (Paper 2 baseline)
    """
    def __init__(self, mode='good', dt=DT):
        self.mode = mode
        self.F,self.H,self.Q,self.R0 = _matrices(dt)
        sc = 0.5*NIS_SOFT
        self.tau_s=NIS_SOFT; self.tau_h=NIS_HARD; self.sc=sc
        self.reset()

    def reset(self):
        self.x = np.zeros(6)
        self.P = np.block([[.25*np.eye(3),np.zeros((3,3))],
                           [np.zeros((3,3)),6.25*np.eye(3)]])
        self.step_count=0; self.coast=0

    def step(self, meas, q):
        q = max(q, Q_MIN)
        xp = self.F@self.x
        Pp = self.F@self.P@self.F.T+self.Q
        Rk = self.R0 if self.mode=='fixed' else self.R0/q
        inn= meas-self.H@xp
        Sk = self.H@Pp@self.H.T+Rk
        nis= float(inn@np.linalg.solve(Sk,inn))
        hard=False; Reff=Rk
        if self.mode=='good' and self.step_count>=10:
            if nis>=self.tau_h and self.coast<5:
                hard=True
            elif nis>self.tau_s:
                w=max(np.exp(-0.5*(nis-self.tau_s)/self.sc),1e-12)
                Reff=Rk/w
        if hard:
            self.x=xp; self.P=Pp; self.coast+=1
        else:
            self.coast=0
            Seff=self.H@Pp@self.H.T+Reff
            K=Pp@self.H.T@np.linalg.solve(Seff,np.eye(3)).T
            self.x=xp+K@inn
            IKH=np.eye(6)-K@self.H
            self.P=IKH@Pp@IKH.T+K@Reff@K.T
        self.step_count+=1
        return self.x.copy(), self.P.copy()

# ─── Digital twin ─────────────────────────────────────────────────
class DigitalTwin:
    def __init__(self,dt=DT):
        sa=SIGMA_ACCEL*0.5
        e3=np.eye(3);z3=np.zeros((3,3))
        F=np.block([[e3,dt*e3],[z3,e3]])
        qp=0.25*dt**4*sa**2;qc=0.5*dt**3*sa**2;qv=dt**2*sa**2
        self.F=F
        self.Q=np.block([[qp*e3,qc*e3],[qc*e3,qv*e3]])
        self.reset()

    def reset(self):
        self.x=np.zeros(6); self.P=np.eye(6)*.25

    def predict(self):
        self.x=self.F@self.x
        self.P=self.F@self.P@self.F.T+self.Q

    def sync(self,xe,Pe): self.x=xe.copy(); self.P=Pe.copy()

    def fuse(self,xe,Pe,w=FUSE_W):
        self.x=w*xe+(1-w)*self.x
        self.P=w*Pe+(1-w)*self.P

# ─── LQR ──────────────────────────────────────────────────────────
def build_lqr(dt=DT):
    e3=np.eye(3);z3=np.zeros((3,3))
    F=np.block([[e3,dt*e3],[z3,e3]])
    B=np.block([[.5*dt**2*e3],[dt*e3]])
    Ql=np.diag([10.,10.,10.,1.,1.,1.]); Rl=np.eye(3)*.1
    try:
        Pi=solve_discrete_are(F,B,Ql,Rl)
        K=np.linalg.solve(Rl+B.T@Pi@B,B.T@Pi@F)
    except Exception:
        K=np.zeros((3,6)); K[:,:3]=2*np.eye(3)
    return K,Ql,Rl

# ─── tr(P_nominal) warmup ─────────────────────────────────────────
def tr_P_nominal_from_warmup(mode='good',dt=DT,n=N_STEPS):
    rng=np.random.default_rng(999)
    pos,vel=make_trajectory(n,dt,rng)
    q=make_quality(n,rng)
    meas=make_measurements(pos,q,rng)
    filt=UpstreamFilter(mode,dt)
    traces=[]
    for k in range(n):
        _,P=filt.step(meas[k],q[k])
        if k>50: traces.append(np.trace(P))
    return float(np.median(traces))

# ─── One trial ────────────────────────────────────────────────────
def one_trial(seed, filter_mode, tr_Pnom, K_lqr, Ql, Rl,
              tau_low=TAU_LOW, tau_high=TAU_HIGH,
              t_period=T_PERIOD, tau_event=TAU_EVENT,
              dt=DT, n=N_STEPS):
    rng=np.random.default_rng(seed)
    pos,vel=make_trajectory(n,dt,rng)
    true_st=np.c_[pos,vel]
    q=make_quality(n,rng)
    qf=np.clip(q*(1+np.random.default_rng(seed+3000).normal(0,.1,n)),Q_MIN,1.)
    meas=make_measurements(pos,q,rng)

    methods=['unconditional','periodic','event','proposed']
    filts={m: UpstreamFilter(filter_mode,dt) for m in methods}
    twins={m: DigitalTwin(dt) for m in methods}

    aoi={m:0 for m in methods}
    hold_streak={m:0 for m in methods}
    sync_ct={m:0 for m in methods}
    fuse_ct={m:0 for m in methods}
    lqg_cum={m:0. for m in methods}
    records={m:[] for m in methods}

    for k in range(n):
        for m in methods:
            twins[m].predict()
            xe,Pe=filts[m].step(meas[k],qf[k])

            # Compute integrity quantities (used by proposed + event)
            trP=np.trace(Pe)
            Ck=qf[k]*np.exp(-ALPHA_CONF*trP/tr_Pnom)
            Dk=float(np.linalg.norm(xe[:3]-twins[m].x[:3]))
            Dr=Dk/(Ck+DELTA_CONF)

            # ── Sync decision ──────────────────────────────────
            if k==0:                       # force init for all
                twins[m].sync(xe,Pe); aoi[m]=0; sync_ct[m]+=1
                action='SYNC'
            elif m=='unconditional':
                twins[m].sync(xe,Pe); aoi[m]=0; sync_ct[m]+=1
                action='SYNC'
            elif m=='periodic':
                if k%t_period==0:
                    twins[m].sync(xe,Pe); aoi[m]=0; sync_ct[m]+=1; action='SYNC'
                else:
                    aoi[m]+=1; action='HOLD'
            elif m=='event':
                if Dk>tau_event:
                    twins[m].sync(xe,Pe); aoi[m]=0; sync_ct[m]+=1; action='SYNC'
                else:
                    aoi[m]+=1; action='HOLD'
            else:   # proposed
                if Dr<=tau_low:
                    action='SYNC'
                elif Dr<=tau_high:
                    action='FUSE'
                else:
                    action='HOLD'
                # anti-spiral guard
                if action=='HOLD' and hold_streak[m]>=MAX_HOLD:
                    action='FUSE'
                if action=='SYNC':
                    twins[m].sync(xe,Pe); aoi[m]=0; sync_ct[m]+=1; hold_streak[m]=0
                elif action=='FUSE':
                    twins[m].fuse(xe,Pe); aoi[m]=0; fuse_ct[m]+=1; hold_streak[m]=0
                else:
                    aoi[m]+=1; hold_streak[m]+=1

            # ── Metrics ────────────────────────────────────────
            terr=float(np.linalg.norm(true_st[k,:3]-twins[m].x[:3]))
            ex=true_st[k]-xe
            eu=(-K_lqr@xe)-(-K_lqr@true_st[k])
            lqg_cum[m]+=float(ex@Ql@ex+eu@Rl@eu)
            records[m].append({'k':k,'t':k*dt,'action':action,
                                'twin_err':terr,'aoi':aoi[m],
                                'Ck':Ck,'Dk':Dk,'Dr':Dr,'q':qf[k]})

    out={}
    for m in methods:
        df=pd.DataFrame(records[m])
        out[m]={
            'rmse_twin': float(np.sqrt(np.mean(df.twin_err**2))),
            'mean_aoi':  float(df.aoi.mean()),
            'max_aoi':   float(df.aoi.max()),
            'lqg_cost':  lqg_cum[m],
            'sync_count':sync_ct[m],
            'fuse_count':fuse_ct[m],
            'df':        df,
        }
    return out

# ─── Monte Carlo ──────────────────────────────────────────────────
def monte_carlo(filter_mode, tr_Pnom, n_seeds=N_SEEDS,
                tau_low=TAU_LOW, tau_high=TAU_HIGH):
    K,Ql,Rl=build_lqr()
    rows=[]
    for s in range(n_seeds):
        res=one_trial(s,filter_mode,tr_Pnom,K,Ql,Rl,tau_low,tau_high)
        for m,v in res.items():
            rows.append({'seed':s,'method':m,'filter':filter_mode,
                         **{k:v[k] for k in v if k!='df'}})
    return pd.DataFrame(rows)

# ─── Pareto sweep ─────────────────────────────────────────────────
def pareto_sweep(filter_mode, tr_Pnom, n_seeds=15):
    """Sweep all methods over their bandwidth-controlling parameter."""
    K,Ql,Rl=build_lqr()
    rows=[]
    # Proposed: sweep tau_high (tau_low fixed at 0.25)
    for th in [0.3,0.5,0.75,1.0,1.5,2.0,3.0,5.0,8.0]:
        for s in range(n_seeds):
            res=one_trial(s,filter_mode,tr_Pnom,K,Ql,Rl,
                          tau_low=0.20,tau_high=th)
            r=res['proposed']
            rows.append({'filter':filter_mode,'method':'proposed',
                         'param':th,'seed':s,
                         'rmse_twin':r['rmse_twin'],'sync_count':r['sync_count'],
                         'mean_aoi':r['mean_aoi']})
    # Periodic: sweep T_period
    for tp in [2,5,8,10,15,20,30,50,80,160,400,800]:
        for s in range(n_seeds):
            res=one_trial(s,filter_mode,tr_Pnom,K,Ql,Rl,t_period=tp)
            r=res['periodic']
            rows.append({'filter':filter_mode,'method':'periodic',
                         'param':tp,'seed':s,
                         'rmse_twin':r['rmse_twin'],'sync_count':r['sync_count'],
                         'mean_aoi':r['mean_aoi']})
    # Event: sweep tau_event (D_k threshold)
    for te in [0.05,0.10,0.15,0.20,0.30,0.50,0.75,1.0,1.5,2.0]:
        for s in range(n_seeds):
            res=one_trial(s,filter_mode,tr_Pnom,K,Ql,Rl,tau_event=te)
            r=res['event']
            rows.append({'filter':filter_mode,'method':'event',
                         'param':te,'seed':s,
                         'rmse_twin':r['rmse_twin'],'sync_count':r['sync_count'],
                         'mean_aoi':r['mean_aoi']})
    # Unconditional: single point
    for s in range(n_seeds):
        res=one_trial(s,filter_mode,tr_Pnom,K,Ql,Rl)
        r=res['unconditional']
        rows.append({'filter':filter_mode,'method':'unconditional',
                     'param':0,'seed':s,
                     'rmse_twin':r['rmse_twin'],'sync_count':r['sync_count'],
                     'mean_aoi':r['mean_aoi']})
    return pd.DataFrame(rows)

# ─── Plotting ─────────────────────────────────────────────────────
COLORS={'proposed':'#2f6fed','periodic':'#EF9F27',
        'event':'#c43c39','unconditional':'#7a828c'}
LABELS={'proposed':'Integrity-aware (proposed)','periodic':'Periodic',
        'event':'Event-triggered','unconditional':'Unconditional'}

def plot_pareto(good_sw, fixed_sw, out_dir):
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    fig.suptitle("Pareto: Twin RMSE vs Sync Events — Two Upstream Filters",
                 fontweight='bold')

    for ax,(sw,title) in zip(axes,[
        (good_sw, "Scenario 1: Good upstream filter (adaptive-R + NIS gate)"),
        (fixed_sw,"Scenario 2: Degraded upstream filter (fixed-R KF)")
    ]):
        for m,col in COLORS.items():
            sub=sw[sw.method==m].groupby('param').agg(
                rmse=('rmse_twin','mean'),
                syncs=('sync_count','mean'),
            ).reset_index().sort_values('syncs')
            if m=='unconditional':
                ax.scatter(sub.syncs,sub.rmse,color=col,
                           marker='*',s=120,zorder=5,label=LABELS[m])
            else:
                ax.plot(sub.syncs,sub.rmse,'o-',color=col,
                        label=LABELS[m],lw=2,ms=5)
        ax.set_xlabel("Mean sync events per trial",fontsize=9)
        ax.set_ylabel("Twin RMSE (m)",fontsize=9)
        ax.set_title(title,fontsize=8.5)
        ax.legend(frameon=False,fontsize=8)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=0.2)

    fig.tight_layout()
    p=out_dir/'fig_pareto_optB.png'
    fig.savefig(p,dpi=200,bbox_inches='tight'); plt.close(fig)
    print(f"  → {p.name}")

def plot_bars(mc_good, mc_fixed, out_dir):
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    fig.suptitle("Twin Divergence, AoI, and LQG Cost — Option B Results",
                 fontweight='bold')
    methods=['proposed','periodic','event','unconditional']
    labels=[LABELS[m] for m in methods]
    colors=[COLORS[m] for m in methods]

    for row,(mc,title) in enumerate([
        (mc_good, "Scenario 1: Good upstream filter"),
        (mc_fixed,"Scenario 2: Degraded upstream filter")
    ]):
        for col,(metric,ylabel) in enumerate([
            ('rmse_twin','Twin RMSE (m)'),
            ('mean_aoi', 'Mean AoI (steps)'),
            ('lqg_cost', 'LQG decision cost'),
        ]):
            ax=axes[row,col]
            means=[float(mc[mc.method==m][metric].mean()) for m in methods]
            stds =[float(mc[mc.method==m][metric].std())  for m in methods]
            xpos=np.arange(len(methods))
            ax.bar(xpos,means,yerr=stds,color=colors,capsize=3,
                   width=0.55,alpha=0.88,edgecolor='white')
            ax.set_xticks(xpos)
            ax.set_xticklabels(labels,fontsize=7,rotation=12,ha='right')
            ax.set_ylabel(ylabel,fontsize=8)
            if col==1: ax.set_title(title,fontsize=8.5)
            ax.spines[['top','right']].set_visible(False)
            ax.grid(axis='y',alpha=0.2)

    fig.tight_layout()
    p=out_dir/'fig_bars_optB.png'
    fig.savefig(p,dpi=200,bbox_inches='tight'); plt.close(fig)
    print(f"  → {p.name}")

def plot_timeline(seed, filter_mode, tr_Pnom, out_dir):
    K,Ql,Rl=build_lqr()
    res=one_trial(seed,filter_mode,tr_Pnom,K,Ql,Rl)
    fig,axes=plt.subplots(4,1,figsize=(12,10),sharex=True)
    tag='good' if filter_mode=='good' else 'fixed'
    fig.suptitle(f"Timeline — {filter_mode} filter — seed {seed}",fontweight='bold')

    for m,col in COLORS.items():
        df=res[m]['df']
        t=df.t
        axes[0].plot(t,df.twin_err.rolling(8,min_periods=1).mean(),
                     color=col,label=LABELS[m],lw=1.6)
        axes[1].plot(t,df.aoi,color=col,lw=1.3)

    prop=res['proposed']['df']
    axes[2].plot(prop.t,prop.q,color='#c43c39',lw=1.5,label='q_k')
    axes[2].plot(prop.t,prop.Ck,color='#2a9d8f',lw=1.2,ls='--',label='C_k')
    axes[2].plot(prop.t,prop.Dr/prop.Dr.max(),color='#2f6fed',
                 lw=1.0,ls=':',label='D_risk (normalised)')

    for action,col in [('SYNC','#2f6fed'),('FUSE','#2a9d8f'),('HOLD','#c43c39')]:
        t_action=prop.loc[prop.action==action,'t']
        if len(t_action):
            axes[3].vlines(t_action,0,1,color=col,alpha=0.25,lw=0.8,label=action)

    axes[0].set_ylabel("Twin err (m)"); axes[0].legend(frameon=False,fontsize=7)
    axes[1].set_ylabel("AoI (steps)")
    axes[2].set_ylabel("Quality / Confidence"); axes[2].legend(frameon=False,fontsize=7)
    axes[3].set_ylabel("Sync actions"); axes[3].set_xlabel("Time (s)")
    axes[3].legend(frameon=False,fontsize=7)
    for ax in axes: ax.spines[['top','right']].set_visible(False); ax.grid(alpha=0.15)

    fig.tight_layout()
    p=out_dir/f'fig_timeline_{tag}_seed{seed}.png'
    fig.savefig(p,dpi=200,bbox_inches='tight'); plt.close(fig)
    print(f"  → {p.name}")

# ─── Summary table ────────────────────────────────────────────────
def print_summary(mc, scenario_label):
    print(f"\n{scenario_label}")
    print(f"{'Method':20} {'TwinRMSE':>10} {'MeanAoI':>9} "
          f"{'LQGcost':>10} {'Syncs':>7} {'Fuses':>7}")
    print("-"*68)
    for m in ['proposed','periodic','event','unconditional']:
        sub=mc[mc.method==m]
        print(f"  {m:18} {sub.rmse_twin.mean():10.4f} "
              f"{sub.mean_aoi.mean():9.2f} "
              f"{sub.lqg_cost.mean():10.1f} "
              f"{sub.sync_count.mean():7.0f} "
              f"{sub.fuse_count.mean():7.0f}")

# ─── Main ─────────────────────────────────────────────────────────
def main():
    out=Path("/home/claude/dt_sync_out_v2"); out.mkdir(exist_ok=True)

    print("Computing tr(P_nominal) for each filter mode...")
    trP_good =tr_P_nominal_from_warmup('good')
    trP_fixed=tr_P_nominal_from_warmup('fixed')
    print(f"  good  filter: tr(P_nominal) = {trP_good:.5f}")
    print(f"  fixed filter: tr(P_nominal) = {trP_fixed:.5f}")

    print("\nMonte Carlo (30 seeds) — good filter...")
    mc_good=monte_carlo('good', trP_good)
    mc_good.to_csv(out/'mc_good.csv',index=False)
    print_summary(mc_good,"Scenario 1: Good upstream filter")

    print("\nMonte Carlo (30 seeds) — fixed filter...")
    mc_fixed=monte_carlo('fixed',trP_fixed)
    mc_fixed.to_csv(out/'mc_fixed.csv',index=False)
    print_summary(mc_fixed,"Scenario 2: Degraded upstream filter")

    print("\nPareto sweep (15 seeds each) — good filter...")
    sw_good=pareto_sweep('good', trP_good,n_seeds=15)
    sw_good.to_csv(out/'pareto_good.csv',index=False)

    print("Pareto sweep (15 seeds each) — fixed filter...")
    sw_fixed=pareto_sweep('fixed',trP_fixed,n_seeds=15)
    sw_fixed.to_csv(out/'pareto_fixed.csv',index=False)

    print("\nPlotting...")
    plot_pareto(sw_good,sw_fixed,out)
    plot_bars(mc_good,mc_fixed,out)
    plot_timeline(0,'good', trP_good, out)
    plot_timeline(0,'fixed',trP_fixed,out)

    print(f"\nAll outputs → {out}")
    return mc_good,mc_fixed,sw_good,sw_fixed

if __name__=="__main__":
    mc_good,mc_fixed,sw_good,sw_fixed=main()
