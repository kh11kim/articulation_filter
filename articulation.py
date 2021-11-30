import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

x_range = np.array([-4,4])
y_range = np.array([-4,4])

skew = lambda x: np.array([[0, -x],[x, 0]])
def se2(s):
    w, v = s[0], s[1:,None]
    return np.block([[skew(s[0]), v],[np.zeros(3)]])
def exp_SE2(s):
    theta, p = s[0], s[1:]
    if theta == 0:
        return np.block([[np.eye(2), p[:,None]],[np.zeros(2), 1]])
    a1, a2 = np.sin(theta)/theta, (1-np.cos(theta))/theta**2
    return np.eye(3) + a1*se2(s) + a2*se2(s)@se2(s)
def exp_SO2(theta):
    return np.cos(theta)*np.eye(2) + np.sin(theta)/theta*skew(theta)
def inv_SE2(T):
    R = T[:2,:2]
    p = T[:2, -1, None]
    return np.block([[R.T, -R.T@p],[np.zeros(2), 1]])
def log_SO2(R):
    return np.arctan2(R[1,0], R[0,0])

def projected_T_rev(T, c, r):
    #inv model
    diff = inv_SE2(c) @ T @ inv_SE2(r)
    theta = np.arctan2(diff[1,0], diff[0,0])
    #fwd model
    That = c @ exp_SE2(np.array([theta,0,0])) @ r
    return That


def draw_SE2(ax, T, color=None):
    origin = T[:2,-1]
    axis_len = 0.2
    xaxis = np.vstack([origin, origin+axis_len*T[:2,0].flatten()])
    yaxis = np.vstack([origin, origin+axis_len*T[:2,1].flatten()])
    if color is None:
        x_color, y_color = 'r', 'g'
    else:
        x_color = y_color = color
    
    ax.plot(*xaxis.T, color=x_color)
    ax.plot(*yaxis.T, color=y_color)

if __name__ == "__main__":
    trans_axis = np.array([np.pi/8, 0, 0])
    x0 = np.array([0, 2, 0])
    T0 = exp_SE2(x0)
    T_ = exp_SE2(trans_axis)
    Ts = [T0]
    T_curr = T0
    for i in range(10):
        T_curr = T_@T_curr
        Ts.append(T_curr)

    #likelihood estimation
    likelihood = 1
    for T in Ts:
        T = T@exp_SE2(np.array([0.1, 0.05, -0.02]))
        c = exp_SE2(np.array([0., 0, 0.1]))
        r = exp_SE2(np.array([0, 2, 0]))
        That = projected_T_rev(T, c, r)
        
        dist_trans = np.linalg.norm(That[:2, -1] - T[:2, -1])
        dist_rot = abs(log_SO2(That) - log_SO2(T))
        dist = [dist_trans, dist_rot]
        p = scipy.stats.multivariate_normal(dist, 0.05*np.eye(2)).pdf(0)
        likelihood *= p
    
    print(dist_trans, dist_rot, p)

    fig, ax = plt.subplots(figsize=[5,5])
    ax.set_aspect(aspect="equal")
    plt.axis([*x_range, *y_range])
    draw_SE2(ax, np.eye(3), 'k')
    for T in Ts:
        draw_SE2(ax, T)