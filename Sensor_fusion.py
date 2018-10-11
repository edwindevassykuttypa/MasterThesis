import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gmplot import gmplot
import webbrowser
from threading import Thread

class CoordinateTransform:  # class for performing coordinate transform

    def __init__(self):
        self.lat = 0.0
        self.a = 6378137.0
        self.b = 6356752.3142
        self.f = (self.a - self.b) / self.a
        self.e_sq = self.f * (2 - self.f)

    def lla_to_ecef(self, lat, log, h):  # Converting Latitude, Longitude, and Altitude to ECEF reference frame.
        Ans = np.array([0.0, 0.0, 0.0])
        Rn = self.a / math.sqrt(1.0 - (self.e_sq * (math.sin(lat) * math.sin(lat))))
        x1 = (Rn + h) * math.cos(lat) * math.cos(log)
        y1 = (Rn + h) * math.cos(lat) * math.sin(log)
        z1 = (Rn * (1.0 - self.e_sq) + h) * math.sin(lat)
        Ans[0] = x1
        Ans[1] = y1
        Ans[2] = z1
        return Ans

    def ecef_to_ned(self, x, y, z, lat0, lon0, h0):  # ECEF to NED Reference frame.

        Ans = np.array([0.0, 0.0, 0.0])
        lamb = lat0
        phi = lon0
        s = math.sin(lamb)
        N = self.a / math.sqrt(1.0 - self.e_sq * s * s)
        sin_lambda = math.sin(lamb)
        cos_lambda = math.cos(lamb)
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        x0 = (h0 + N) * cos_lambda * cos_phi
        y0 = (h0 + N) * cos_lambda * sin_phi
        z0 = (h0 + (1.0 - self.e_sq) * N) * sin_lambda
        xd = x - x0
        yd = y - y0
        zd = z - z0
        xEast = -sin_phi * xd + cos_phi * yd
        yNorth = -cos_phi * sin_lambda * xd - sin_lambda * sin_phi * yd + cos_lambda * zd
        zUp = cos_lambda * cos_phi * xd + cos_lambda * sin_phi * yd + sin_lambda * zd
        Ans[0] = yNorth
        Ans[1] = xEast
        Ans[2] = zUp
        return Ans

    def lla_to_ned(self, lat, log, alt, lat0, log0, alt0):  # lat log alt to north east down
        Ans = np.array([0.0, 0.0, 0.0])
        Ans[0], Ans[1], Ans[2] = self.lla_to_ecef(lat, log, alt)
        Ans[0], Ans[1], Ans[2] = self.ecef_to_ned(Ans[0], Ans[1], Ans[2], lat0, log0, alt0)
        return Ans

    def enu_to_uvw(self, north, east, up, lat0, lon0):  # north east down to azimuth elevation inclination
        Ans = np.array([0.0, 0.0, 0.0])
        t = math.cos(lat0) * up - math.sin(lat0) * north
        w = math.sin(lat0) * up + math.cos(lat0) * north
        u = math.cos(lon0) * t - math.sin(lon0) * east
        v = math.sin(lon0) * t + math.cos(lon0) * east
        Ans[0] = u
        Ans[1] = v
        Ans[2] = w
        return Ans

    def ned_to_ecef(self, north, east, down, lat0, log0, alt0):  # north east down to earth centered earth fixed frame
        Ans = np.array([0.0, 0.0, 0.0])
        x0, y0, z0 = self.lla_to_ecef(lat0, log0, alt0)
        dx, dy, dz = self.enu_to_uvw(north, east, down, lat0, log0)
        Ans[0] = x0 + dx
        Ans[1] = y0 + dy
        Ans[2] = z0 + dz
        return Ans

    def ecef_to_lla(self, x, y, z):  # ECEF to Latitude, Longitude, Altitude reference frame.
        Ans = np.array([0.0, 0.0, 0.0])
        esq = 6.69437999014 * 0.001
        e1sq = 6.73949674228 * 0.001
        r = math.hypot(x, y)
        Esq = self.a ** 2.0 - self.b ** 2.0
        F = 54.0 * self.b ** 2.0 * z ** 2.0
        G = r ** 2.0 + (1.0 - esq) * z ** 2.0 - esq * Esq
        C = (esq ** 2.0 * F * r ** 2.0) / (pow(G, 3))
        S = math.pow((1.0 + C + math.sqrt(C ** 2.0 + 2.0 * C)), 1.0 / 3.0)
        P = F / (3.0 * pow((S + 1 / S + 1), 2) * G ** 2.0)
        Q = math.sqrt(1.0 + 2.0 * esq ** 2 * P)
        r_0 = -(P * esq * r) / (1 + Q) + math.sqrt(
            0.5 * self.a ** 2 * (1.0 + 1.0 / Q) - P * (1 - esq) * z ** 2 / (Q * (1 + Q)) - 0.5 * P * r ** 2)
        U = math.sqrt(pow((r - esq * r_0), 2) + z ** 2)
        V = math.sqrt(pow((r - esq * r_0), 2) + (1.0 - esq) * z ** 2)
        Z_0 = self.b ** 2.0 * z / (self.a * V)
        al = U * (1.0 - self.b ** 2.0 / (self.a * V))
        la = math.atan((z + e1sq * Z_0) / r)
        lo = math.atan2(y, x)
        Ans[0] = la
        Ans[1] = lo
        Ans[2] = al
        return Ans

    def ned_to_lla(self, north, east, down, lat0, log0, alt0):  # north east down to lat log alt frame
        Ans = np.array([0.0, 0.0, 0.0])
        Ans[0], Ans[1], Ans[2] = self.ned_to_ecef(north, east, down, lat0, log0, alt0)
        Ans[0], Ans[1], Ans[2] = self.ecef_to_lla(Ans[0], Ans[1], Ans[2])
        return Ans


class GetData:

    def __init__(self):
        self.Time = []
        self.Lat = []
        self.Long = []
        self.Alt = []
        self.GPS_accuracy = []
        self.acc_x = []
        self.acc_y = []
        self.acc_z = []
        print (" Initialized ")

    def read_csv(self):  # function to read data from a csv file

        df = pd.read_csv("arun.csv")
        num_py_array = df.as_matrix()
        for i in num_py_array:
            self.Time.append(i[29])
            self.Lat.append(i[21])
            self.Long.append(i[22])
            self.Alt.append(i[23])
            self.GPS_accuracy.append(i[26])
            self.acc_x.append(i[7])
            self.acc_y.append(i[8])
            self.acc_z.append(i[9])
        for i in range(len(self.Time)):
            self.Lat[i] = self.Lat[i] * math.pi / 180.0
            self.Long[i] = self.Long[i] * math.pi / 180.0


class KalmanFilter:  # Kalman filter

    def prior(self, X, F, P, Q, phi):  # prediction step
        F_t = F.T
        X_new = np.dot(F, X)
        #print(X_new.shape)
        P_new = np.dot(np.dot(F, P), F_t) + np.dot(Q, phi)
        return X_new, P_new

    def update(self, X, P, Z, R, H):  # update step
        H_t = H.T
        #print(Z.shape)
        Y = Z - np.dot(H, X)  # residual calculation
        K = np.dot(np.dot(P, H_t), np.linalg.inv(np.dot(np.dot(H, P), H_t) + R))  # finding kalman gain
        #print(K.shape , Y.shape)
        X_new = X + np.dot(K, Y)  # computing new state variable
        p_new_half = np.dot(np.dot(K, R), K.T)
        P_new = np.dot((np.dot((np.identity(6) - np.dot(K, H)), P)), (np.identity(6) - np.dot(K, H)).T) + p_new_half  # new state covarience
        return X_new, P_new


class PlotData:

    def plot_results(self, lat, log, lat_in, log_in ):

        plt.figure(1)
        plt.plot(log_in, lat_in, 'g', label='GPS readings in LLA frame')
        plt.plot(Log, Lat, 'r', label='O/P of Kalman Filter in LLA frame')
        x = plt.ylabel('Latitude')
        y = plt.xlabel('Longitude')
        plt.title('Robot_Position_Estimation')
        # plt.annotate("Karl-Wilhelm-Platz", (lon_calculated[0],lat_calculated[0]), ha="center", va="center", bbox=dict(boxstyle="round", fc="w"))
        # plt.annotate("Kronenplatz", (lon_calculated[len(lon_calculated)-1], lat_calculated[len(lon_calculated)-1]), ha="center", va="center", bbox=dict(boxstyle="round", fc="w"))
        plt.legend()
        plt.show()


    def plot_results_in_browser(self, lat, log):

        gmap = gmplot.GoogleMapPlotter(lat[int(len(lat) / 2)], log[int(len(log) / 2)], 17)
        gmap.plot(lat, log, 'red', edge_width=3)
        gmap.draw("my_map.html")
        url = "my_map.html"
        new = 2
        vivaldidir = "/usr/bin/chromium-browser %s"
        webbrowser.get(vivaldidir).open(url, new=new)

gd = GetData()
ct = CoordinateTransform()
gd.read_csv()
kf = KalmanFilter()
gm = PlotData()
North = [0.0 for _ in range(len(gd.Time))]
East = [0.0 for _ in range(len(gd.Time))]
Down = [0.0 for _ in range(len(gd.Time))]
Lat = [0.0 for _ in range(len(gd.Time))]
Log = [0.0 for _ in range(len(gd.Time))]
Alt = [0.0 for _ in range(len(gd.Time))]
v_north = 0.0
v_east = 0.0
past_time = 509
file_op = open("op_data.txt", "w+")
file_test = open("test_data.txt", "w+")
lat_input = []
log_input = []

X = np.array([[0.0],  # x_pos
              [0.0],  # y_pos
              [0.0],  # x_vel
              [0.0],  # y_vel
              [0.0],  # x_acc
              [0.0]])  # y_acc

P = np.diag([5.0, 5.0, 500., 500., 500.0, 500.0])

phi = 0.000001

H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])


def matrix_initialization(dt, accuracy):

    F = np.array([[1.0, 0.0, dt, 0.0, 0.5 * dt * dt, 0.0],
                  [0.0, 1.0, 0.0, dt, 0.0, 0.5 * dt * dt],
                  [0.0, 0.0, 1.0, 0.0, dt, 0.0],
                  [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    Q = np.array([[(pow(dt, 5) / 20.0), 0.0, (pow(dt, 4) / 8.0), 0.0, (pow(dt, 3) / 6.0), 0.0],
                  [0.0, pow(dt, 5) / 20.0, 0.0, (pow(dt, 4) / 8.0), 0.0, (pow(dt, 3) / 6.0)],
                  [(pow(dt, 4) / 8.0), 0.0, (pow(dt, 3) / 3.0), 0.0, (pow(dt, 2) / 2.0), 0.0],
                  [0.0, (pow(dt, 4) / 8.0), 0.0, (pow(dt, 3) / 3.0), 0.0, (pow(dt, 2) / 2.0)],
                  [(pow(dt, 3) / 6.0), 0.0, (pow(dt, 2) / 2.0), 0.0, dt, 0.0],
                  [0.0, (pow(dt, 3) / 6.0), 0.0, (pow(dt, 2) / 2.0), 0.0, dt]])

    R = np.array([[pow(accuracy, 2), 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, pow(accuracy, 2), 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 4.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    return F, Q, R


for i in range(len(gd.Time)):
    dt = abs(gd.Time[i] - past_time) / 1000.0  # calculating time difference between measurements
    North[i], East[i], Down[i] = ct.lla_to_ned(gd.Lat[i], gd.Long[i], gd.Alt[i], gd.Lat[0], gd.Long[0], gd.Alt[0])
    # converting radians to degree
    lat_input.append(gd.Lat[i] * 180 / math.pi)
    log_input.append(gd.Long[i] * 180/math.pi)
    # file operation to save the input data
    file_test.write(str(gd.Lat[i] * 180 / math.pi) + "     " + str(gd.Long[i] * 180/math.pi) + "\n")
    # calculating  the velocity from acceleration values
    v_north = v_north + dt * gd.acc_y[i]
    v_east = v_east + dt * gd.acc_x[i]
    # creating the measurement matrix for kalman filter
    Z = np.array([[North[i]], [East[i]], [v_north], [v_east], [gd.acc_y[i]], [gd.acc_x[i]]])
    # extracting gps accuracy from readings
    accuracy = gd.GPS_accuracy[i]
    # initialization of F,Q,R matrices
    F, Q, R = matrix_initialization(dt, accuracy)
    # kalman filter steps

    X, P = kf.prior(X, F, P, Q, phi)

    X, P = kf.update(X, P, Z, R, H)

    # converting kalman filter output to lat, log, alt values
    Lat[i], Log[i], Alt[i] = ct.ned_to_lla(X.item(0), X.item(1), Down[i], gd.Lat[0], gd.Long[0], gd.Alt[0])
    # converting radians to degrees
    Lat[i] = Lat[i]*180 / math.pi
    Log[i] = Log[i] * 180 / math.pi
    # file operations for saving output file
    file_op.write(str(Lat[i]) + "     " + str(Log[i]) + "\n")
    past_time = gd.Time[i]


file_test.close()
file_op.close()

Thread(target = gm.plot_results_in_browser(Lat, Log)).start()
Thread(target = gm.plot_results(Lat, Log, lat_input, log_input)).start()


