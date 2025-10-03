# analyze_signal.py
import numpy as np

class AnalyzeSignal:
    def __init__(self, signal, normalize=True):
        signal = np.array(signal)
        self.x = signal[:,0]
        self.y = signal[:,1]

        if normalize:
            y_min = np.min(self.y)
            y_max = np.max(self.y)
            self.y_norm = (self.y - y_min) / (y_max - y_min)
        else:
            self.y_norm = self.y.copy()
        
        self.dy = np.gradient(self.y_norm, self.x)
        self.d2y = np.gradient(self.dy, self.x)
        
        self._get_max_global_point()
        self._get_critical_points()
        self._get_inflexion_points()

    
    def _get_max_global_point(self):
        idx = np.argmax(np.abs(self.y_norm))
        self.max_point = (self.x[idx], self.y_norm[idx])

    def _get_critical_points(self):
        # puntos donde cambia el signo de la primera derivada
        idx = np.where(np.diff(np.sign(self.dy)) != 0)[0]

        # segunda derivada aproximada
        d2y = np.gradient(self.dy, self.x)

        crit_points = []
        for i in idx:
            if d2y[i] < 0:
                crit_type = "max"
            elif d2y[i] > 0:
                crit_type = "min"
            else:
                crit_type = "flat"
            crit_points.append((self.x[i], self.y_norm[i], crit_type))
        
        self.crit_points = np.array(crit_points, dtype=object)
        return self.crit_points

    def _get_inflexion_points(self):
        idx = np.where(np.diff(np.sign(self.d2y)) != 0)[0]
        self.infl_points = np.column_stack((self.x[idx], self.y_norm[idx]))
    
    def get_max_point(self):
        return self.max_point
    
    def get_critical_points(self):
        return self.crit_points
    
    def get_inflexion_points(self):
        return self.infl_points
    
    def get_normalized_signal(self):
        return np.column_stack((self.x, self.y_norm))