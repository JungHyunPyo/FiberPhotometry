# Fiber Photometry GUI — Doric + TTL (digital-only) with min-interval option + PSTH
import os, sys, math, tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    import h5py
except Exception:
    h5py = None

try:
    from scipy.optimize import curve_fit
except Exception:
    curve_fit = None

# ---------- Helpers ----------
def poly_bleach_correction(signal, time, order=2):
    s = np.asarray(signal, dtype=float)
    t = np.asarray(time, dtype=float)
    if order <= 0 or s.size < 5:
        return s.copy()
    tnorm = (t - t.min()) / (t.max() - t.min() + 1e-9)
    try:
        coeff = np.polyfit(tnorm, s, int(order))
        trend = np.polyval(coeff, tnorm)
        return s - trend + np.median(trend)
    except Exception:
        return s.copy()

def double_exp_decay(x, a1, tau1, a2, tau2, c):
    x = np.asarray(x, dtype=float)
    return a1 * np.exp(-x / max(tau1, 1e-9)) + a2 * np.exp(-x / max(tau2, 1e-9)) + c

def bleach_correct(signal, time, method="Polynomial", poly_order=2, fit_start_s=None, fit_end_s=None):
    s = np.asarray(signal, dtype=float)
    t = np.asarray(time, dtype=float)
    if method.lower().startswith("poly"):
        return poly_bleach_correction(s, t, order=poly_order)
    if curve_fit is None:
        raise RuntimeError("Double-exponential requires SciPy. Please install scipy.")
    if fit_start_s is None: fit_start_s = float(t.min())
    if fit_end_s is None: fit_end_s = float(t.max())
    mask = (t >= fit_start_s) & (t <= fit_end_s)
    if not np.any(mask):
        mask = np.ones_like(t, dtype=bool)
    tx = t[mask] - t[mask][0]
    sx = s[mask]
    if sx.size < 10:
        return s.copy()
    a1 = float(np.percentile(sx, 90) - np.percentile(sx, 10))
    a2 = a1 * 0.3
    tau1 = max((tx.max() - tx.min()) / 10.0, 1.0)
    tau2 = max((tx.max() - tx.min()) / 2.0, 5.0)
    c0 = float(np.median(sx))
    p0 = [a1, tau1, a2, tau2, c0]
    bounds = ([-np.inf, 0.1, -np.inf, 0.5, -np.inf], [np.inf, 1e6, np.inf, 1e7, np.inf])
    try:
        popt, _ = curve_fit(double_exp_decay, tx, sx, p0=p0, bounds=bounds, maxfev=20000)
        fitted = double_exp_decay(t - t[mask][0], *popt)
    except Exception:
        return s.copy()
    return s - fitted + np.median(fitted)

def auto_sampling_rate(time):
    t = np.asarray(time, dtype=float)
    diffs = np.diff(t)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return None
    dt = np.median(diffs)
    if dt <= 0:
        return None
    return 1.0 / dt

def enforce_min_interval(times, min_interval):
    """Keep first event, drop subsequent events within min_interval seconds."""
    times = np.asarray(times, dtype=float)
    if times.size == 0:
        return times
    times_sorted = np.sort(times)
    kept = [times_sorted[0]]
    last = times_sorted[0]
    for t in times_sorted[1:]:
        if (t - last) >= min_interval:
            kept.append(t)
            last = t
    return np.array(kept, dtype=float)

def interp_align(signal, time, onsets, pre, post, sampling_rate=None):
    """Interpolate to a common grid for each onset; returns trials x samples array and the relative time grid."""
    t = np.asarray(time, dtype=float)
    s = np.asarray(signal, dtype=float)
    if sampling_rate is None:
        sampling_rate = auto_sampling_rate(t)
    if sampling_rate is None or sampling_rate <= 0:
        sampling_rate = 50.0  # safe fallback
    step = 1.0 / sampling_rate
    rel = np.arange(-pre, post, step, dtype=float)
    X = []
    for o in onsets:
        target = o + rel
        y = np.interp(target, t, s, left=np.nan, right=np.nan)
        if np.isnan(y).any():
            # if too many nans (out of range), skip this trial
            if np.mean(np.isnan(y)) > 0.05:
                continue
            # otherwise, fill small edge NaNs with nearest valid
            first = np.argmax(~np.isnan(y))
            last = len(y) - 1 - np.argmax(~np.isnan(y[::-1]))
            if first < last:
                y[:first] = y[first]
                y[last+1:] = y[last]
            y = np.nan_to_num(y, nan=np.nanmean(y))
        X.append(y)
    return (np.array(X), rel)

def dff_from_baseline_mean(aligned, baseline_frames):
    arr = np.asarray(aligned, dtype=float)
    base = arr[:, baseline_frames]
    f0 = np.mean(base, axis=1, keepdims=True)
    f0 = np.where(np.abs(f0) < 1e-12, 1e-12, f0)
    return (arr - f0) / f0

def zscore_by_baseline(traces, baseline_frames):
    traces = np.asarray(traces, dtype=float)
    base = traces[:, baseline_frames]
    mu = np.mean(base, axis=1, keepdims=True)
    sd = np.std(base, axis=1, keepdims=True) + 1e-8
    return (traces - mu) / sd

# ---------- GUI ----------
class FPApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fiber Photometry Toolkit")
        self.geometry("1200x820")

        # Data holders
        self.df_analog = None     # DataFrame with analog signals (common time axis)
        self.analog_time_col = None
        self.digital = {}         # dict: name -> (time_array, value_array)
        self.onsets = []
        self.corrected_selected = None  # corrected full-length signal used for overview/PSTH
        self.rel_grid = None
        self.aligned_mat = None

        # Params
        self.pre_s = tk.DoubleVar(value=5.0)
        self.post_s = tk.DoubleVar(value=10.0)
        self.bleach_order = tk.IntVar(value=2)
        self.bleach_method = tk.StringVar(value="Polynomial")
        self.fit_start_s = tk.DoubleVar(value=0.0)
        self.fit_end_s = tk.DoubleVar(value=0.0)
        self.use_red = tk.BooleanVar(value=False)
        self.signal_choice = tk.StringVar(value="Green")
        self.use_zscore = tk.BooleanVar(value=True)
        self.min_ttl_interval = tk.DoubleVar(value=1.0)  # seconds

        self._build_ui()

    # ---------- UI ----------
    
    def _init_window(self):
        # Start maximized where possible
        try:
            self.state('zoomed')  # Windows
        except Exception:
            try:
                self.attributes('-zoomed', True)  # Some X11/Win variants
            except Exception:
                sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
                self.geometry(f"{sw}x{sh}+0+0")
        # Fullscreen toggle helpers
        self.bind("<F11>", self._toggle_fullscreen)
        self.bind("<Escape>", lambda e: self.attributes("-fullscreen", False))
        # Simple menubar with View -> Fullscreen
        menubar = tk.Menu(self)
        viewmenu = tk.Menu(menubar, tearoff=0)
        viewmenu.add_command(label="Toggle Fullscreen (F11)", command=self._toggle_fullscreen)
        viewmenu.add_command(label="Exit Fullscreen (Esc)", command=lambda: self.attributes("-fullscreen", False))
        menubar.add_cascade(label="View", menu=viewmenu)
        self.config(menu=menubar)

    def _toggle_fullscreen(self, event=None):
        try:
            cur = bool(self.attributes("-fullscreen"))
        except Exception:
            cur = False
        self.attributes("-fullscreen", not cur)
    
    def _build_ui(self):
        # File bar
        fb = ttk.Frame(self); fb.pack(fill="x", padx=10, pady=8)
        ttk.Button(fb, text="Open CSV...", command=self.open_csv).pack(side="left")
        ttk.Button(fb, text="Open H5...", command=self.open_h5).pack(side="left", padx=(8,0))
        ttk.Button(fb, text="Open DORIC...", command=self.open_doric).pack(side="left", padx=(8,0))
        self.file_label = ttk.Label(fb, text="No file loaded"); self.file_label.pack(side="left", padx=10)

        # Mapping / TTL
        mapf = ttk.LabelFrame(self, text="Mapping & TTL"); mapf.pack(fill="x", padx=10, pady=8)
        self.cb_time = ttk.Combobox(mapf, state="readonly")
        self.cb_green = ttk.Combobox(mapf, state="readonly")
        self.cb_red = ttk.Combobox(mapf, state="readonly")
        ttk.Label(mapf, text="Time:").grid(row=0, column=0, sticky="e", padx=5, pady=4)
        self.cb_time.grid(row=0, column=1, sticky="we", padx=5, pady=4)
        ttk.Label(mapf, text="Green:").grid(row=0, column=2, sticky="e", padx=5, pady=4)
        self.cb_green.grid(row=0, column=3, sticky="we", padx=5, pady=4)
        ttk.Checkbutton(mapf, text="Use Red", variable=self.use_red, command=self._toggle_red).grid(row=1, column=0, sticky="w", padx=5, pady=4)
        ttk.Label(mapf, text="Red:").grid(row=1, column=2, sticky="e", padx=5, pady=4)
        self.cb_red.grid(row=1, column=3, sticky="we", padx=5, pady=4)

        ttk.Label(mapf, text="TTL source (Digital only):").grid(row=2, column=0, sticky="e", padx=5, pady=4)
        self.cb_ttl_source = ttk.Combobox(mapf, state="readonly")
        self.cb_ttl_source.grid(row=2, column=1, sticky="we", padx=5, pady=4)
        ttk.Label(mapf, text="Min TTL interval (s):").grid(row=2, column=2, sticky="e", padx=5, pady=4)
        ttk.Entry(mapf, textvariable=self.min_ttl_interval, width=10).grid(row=2, column=3, sticky="w", padx=5, pady=4)

        for i in range(4): mapf.grid_columnconfigure(i, weight=1)

        # Parameters
        param = ttk.LabelFrame(self, text="Parameters"); param.pack(fill="x", padx=10, pady=8)
        ttk.Label(param, text="Pre (s):").grid(row=0, column=0, sticky="e", padx=5, pady=4)
        ttk.Entry(param, textvariable=self.pre_s, width=8).grid(row=0, column=1, sticky="w", padx=5, pady=4)
        ttk.Label(param, text="Post (s):").grid(row=0, column=2, sticky="e", padx=5, pady=4)
        ttk.Entry(param, textvariable=self.post_s, width=8).grid(row=0, column=3, sticky="w", padx=5, pady=4)
        ttk.Label(param, text="Bleach:").grid(row=1, column=0, sticky="e", padx=5, pady=4)
        ttk.Combobox(param, state="readonly", values=["Polynomial","Double-exponential"], textvariable=self.bleach_method, width=18).grid(row=1, column=1, sticky="w", padx=5, pady=4)
        ttk.Label(param, text="Poly order (0-3):").grid(row=1, column=2, sticky="e", padx=5, pady=4)
        ttk.Entry(param, textvariable=self.bleach_order, width=8).grid(row=1, column=3, sticky="w", padx=5, pady=4)
        ttk.Label(param, text="Fit window (s):").grid(row=2, column=0, sticky="e", padx=5, pady=4)
        ttk.Entry(param, textvariable=self.fit_start_s, width=8).grid(row=2, column=1, sticky="w", padx=5, pady=4)
        ttk.Label(param, text="to").grid(row=2, column=2, sticky="e", padx=5, pady=4)
        ttk.Entry(param, textvariable=self.fit_end_s, width=8).grid(row=2, column=3, sticky="w", padx=5, pady=4)
        ttk.Label(param, text="Signal:").grid(row=3, column=0, sticky="e", padx=5, pady=4)
        self.cmb_signal = ttk.Combobox(param, state="readonly", values=["Green","Red"], textvariable=self.signal_choice, width=18)
        self.cmb_signal.grid(row=3, column=1, sticky="w", padx=5, pady=4)
        ttk.Checkbutton(param, text="Z-score after dF/F (pre→0s mean)", variable=self.use_zscore).grid(row=3, column=2, columnspan=2, sticky="w", padx=5, pady=4)

        # Actions
        act = ttk.Frame(self); act.pack(fill="x", padx=10, pady=8)
        ttk.Button(act, text="Fit & TTL detection", command=self.fit_and_detect).pack(side="left")
        ttk.Button(act, text="Run PSTH", command=self.run_psth).pack(side="left", padx=(8,0))
        ttk.Button(act, text="Save aligned CSV", command=self.save_aligned).pack(side="left", padx=(8,0))

        # Plots
        plots = ttk.LabelFrame(self, text="Overview & PSTH"); plots.pack(fill="both", expand=True, padx=10, pady=8)
        self.fig_over = Figure(figsize=(6.0, 3.2), dpi=100); self.ax_over = self.fig_over.add_subplot(111)
        self.canvas_over = FigureCanvasTkAgg(self.fig_over, master=plots); self.canvas_over.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.fig_psth = Figure(figsize=(6.2, 6.6), dpi=100)
        self.ax_mean = self.fig_psth.add_subplot(211)
        self.ax_heat = self.fig_psth.add_subplot(212, sharex=self.ax_mean)
        self.canvas_psth = FigureCanvasTkAgg(self.fig_psth, master=plots)
        self.canvas_psth.get_tk_widget().pack(side="top", fill="both", expand=True)

        self._toggle_red()

    # ---------- Data loading ----------
    def _toggle_red(self):
        self.cb_red.config(state=("readonly" if self.use_red.get() else "disabled"))
        # update signal choices
        choices = ["Green"]
        if self.use_red.get():
            choices.append("Red")
        self.cmb_signal["values"] = choices
        if self.signal_choice.get() not in choices:
            self.signal_choice.set(choices[0])

    def _set_analog_df(self, time, series_dict, file_name):
        # Build a DataFrame on a common time axis, resampling series as needed
        df = pd.DataFrame({"Time": np.asarray(time, dtype=float)})
        for name, (sig, stime) in series_dict.items():
            sig = np.asarray(sig, dtype=float)
            if stime is not None and len(stime) == len(sig):
                y = np.interp(df["Time"].values, np.asarray(stime, dtype=float), sig, left=np.nan, right=np.nan)
                y = np.nan_to_num(y, nan=np.nanmean(sig) if np.isfinite(np.nanmean(sig)) else 0.0)
            else:
                # fallback: truncate or pad
                y = np.full(len(df), np.nan)
                n = min(len(sig), len(df))
                y[:n] = sig[:n]
                y = np.nan_to_num(y, nan=np.nanmean(sig) if np.isfinite(np.nanmean(sig)) else 0.0)
            df[name] = y
        self.df_analog = df
        self.analog_time_col = "Time"
        self.file_label.config(text=file_name)
        cols = list(df.columns)
        # Populate mapping lists
        for cb in [self.cb_time, self.cb_green, self.cb_red]:
            cb["values"] = cols
        self.cb_time.set("Time")
        # heuristic for green/red
        lower = {c.lower(): c for c in cols}
        def pick(*keys):
            for k in keys:
                for lc, orig in lower.items():
                    if k in lc:
                        return orig
            return None
        g = pick("465", "gcamp", "green", "lock", "analog in 0", "ain01", "a_in_0", "signal demodulated")
        if g: self.cb_green.set(g)
        r = pick("405", "560", "isosbestic", "red", "analog in 1", "ain02", "a_in_1")
        if r: self.cb_red.set(r)

    def open_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV","*.csv"),("All","*.*")])
        if not path: return
        try:
            df = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("CSV", f"Read failed: {e}"); return
        # No digital in CSV path in this simplified loader
        self.digital = {}
        # Ensure a Time column exists
        if "Time" not in df.columns:
            # try 'time' or first column
            for c in df.columns:
                if c.lower().startswith("time"):
                    df = df.rename(columns={c: "Time"})
                    break
            if "Time" not in df.columns:
                df.insert(0, "Time", np.arange(len(df), dtype=float))
        series_dict = {c: (df[c].values, None) for c in df.columns if c != "Time"}
        self._set_analog_df(df["Time"].values, series_dict, os.path.basename(path))
        self._populate_ttl_sources()

    def open_h5(self):
        if h5py is None:
            messagebox.showwarning("H5", "h5py not installed."); return
        path = filedialog.askopenfilename(filetypes=[("HDF5","*.h5 *.hdf5"),("All","*.*")])
        if not path: return
        self._load_doric_like(path, label=os.path.basename(path))

    def open_doric(self):
        if h5py is None:
            messagebox.showwarning("Doric", "h5py not installed."); return
        path = filedialog.askopenfilename(filetypes=[("Doric HDF5","*.doric"),("All","*.*")])
        if not path: return
        self._load_doric_like(path, label=os.path.basename(path))

    def _load_doric_like(self, path, label="loaded"):
        try:
            with h5py.File(path, "r") as f:
                # Collect datasets
                datasets = {}
                def visit(name, obj):
                    import h5py as _h
                    if isinstance(obj, _h.Dataset) and obj.ndim == 1:
                        try:
                            arr = obj[()]
                            if np.asarray(arr).dtype.kind in "iuf":  # numeric
                                datasets[name] = np.asarray(arr)
                        except Exception:
                            pass
                f.visititems(lambda name, obj: visit(name, obj))

            # Separate digital and candidate analog
            # Digital time
            dig_times = [k for k in datasets.keys() if "digital" in k.lower() and "time" in k.lower()]
            if not dig_times:
                dig_times = [k for k in datasets.keys() if "dio" in k.lower() and "time" in k.lower()]
            digital_channels = [k for k in datasets.keys() if ("digital" in k.lower() or "dio" in k.lower()) and "time" not in k.lower()]
            self.digital = {}
            if dig_times:
                tkey = sorted(dig_times, key=lambda x: datasets[x].size, reverse=True)[0]
                tdig = datasets[tkey]
                for ch in digital_channels:
                    self.digital[ch] = (tdig, datasets[ch])

            # Analog: choose a main time axis (largest non-digital 'time')
            analog_time_keys = [k for k in datasets.keys() if "time" in k.lower() and "digital" not in k.lower() and "dio" not in k.lower()]
            if analog_time_keys:
                at_key = sorted(analog_time_keys, key=lambda x: datasets[x].size, reverse=True)[0]
                t_analog = datasets[at_key]
            else:
                # fallback: use digital time if exists
                t_analog = None
                if self.digital:
                    any_dig_time = next(iter(self.digital.values()))[0]
                    t_analog = any_dig_time
                else:
                    # last fallback: create an index-based time
                    any_key = max(datasets, key=lambda x: datasets[x].size)
                    n = datasets[any_key].size
                    t_analog = np.arange(n, dtype=float)

            # Build analog series dict; try to pair signals with their own time when available
            series = {}
            for name, arr in datasets.items():
                lname = name.lower()
                if "time" in lname: 
                    continue
                if ("digital" in lname) or ("dio" in lname):
                    continue
                # Try to find sibling time in same folder
                # e.g., "/A/B/C/Signal" -> check "/A/B/C/Time"
                parts = name.split("/")
                sib_time = None
                if len(parts) > 1:
                    sib = "/".join(parts[:-1] + ["Time"])
                    if sib in datasets and datasets[sib].size == arr.size:
                        sib_time = datasets[sib]
                series[name] = (arr, sib_time)

            self._set_analog_df(t_analog, series, label)
            self._populate_ttl_sources()

        except Exception as e:
            messagebox.showerror("Doric/H5", f"Load failed: {e}")

    def _populate_ttl_sources(self):
        # Only digital sources are listed
        names = list(self.digital.keys())
        self.cb_ttl_source["values"] = names
        if names:
            self.cb_ttl_source.set(names[0])
        else:
            self.cb_ttl_source.set("")

    # ---------- TTL & analysis ----------
    def _mapping(self):
        if self.df_analog is None:
            messagebox.showwarning("Data", "Load data first."); return None
        tcol = self.cb_time.get().strip()
        gcol = self.cb_green.get().strip()
        rcol = self.cb_red.get().strip() if self.use_red.get() else None
        if not tcol or not gcol:
            messagebox.showwarning("Mapping", "Please select Time and Green columns."); return None
        return tcol, gcol, rcol

    def fit_and_detect(self):
        m = self._mapping()
        if m is None: return
        tcol, gcol, rcol = m
        time = self.df_analog[tcol].values.astype(float)
        g = self.df_analog[gcol].values.astype(float)
        r = self.df_analog[rcol].values.astype(float) if (self.use_red.get() and rcol) else None

        # Bleach correction on the chosen signal for overview
        method = self.bleach_method.get()
        fs, fe = float(self.fit_start_s.get()), float(self.fit_end_s.get())
        if fe <= fs:
            fs, fe = float(time.min()), float(time.max())
        try:
            g_corr = bleach_correct(g, time, method=method, poly_order=max(0, min(int(self.bleach_order.get()),3)), fit_start_s=fs, fit_end_s=fe)
        except RuntimeError as e:
            messagebox.showerror("Bleach", str(e)); return
        r_corr = None
        if r is not None:
            try:
                r_corr = bleach_correct(r, time, method=method, poly_order=max(0, min(int(self.bleach_order.get()),3)), fit_start_s=fs, fit_end_s=fe)
            except RuntimeError:
                r_corr = r.copy()

        sig = g_corr if (self.signal_choice.get() != "Red" or r_corr is None) else r_corr
        self.corrected_selected = sig  # save for PSTH

        # TTL detection from digital source
        src = self.cb_ttl_source.get().strip()
        if not src or src not in self.digital:
            messagebox.showwarning("TTL", "No digital TTL source selected."); return
        tdig, dig = self.digital[src]
        thr = np.nanmedian(dig)
        binv = (dig > thr).astype(int)
        edges = np.where(np.diff(binv) == 1)[0]
        onsets = tdig[edges + 1] if edges.size > 0 else np.array([])
        # enforce min interval
        onsets = enforce_min_interval(onsets, float(self.min_ttl_interval.get()))
        self.onsets = list(map(float, onsets))

        # Overview plot with TTL windows
        self.ax_over.clear()
        self.ax_over.plot(time, sig, linewidth=1.0)
        pre = float(self.pre_s.get()); post = float(self.post_s.get())
        for o in self.onsets:
            self.ax_over.axvspan(o - pre, o + post, alpha=0.15)
            self.ax_over.axvline(o, linestyle="--", linewidth=1)
        self.ax_over.set_xlabel("Time (s)"); self.ax_over.set_ylabel("Corrected signal")
        self.ax_over.set_title(f"Overview ({len(self.onsets)} TTLs, min interval {self.min_ttl_interval.get():.2f}s)")
        self.canvas_over.draw()
        messagebox.showinfo("TTL", f"Detected {len(self.onsets)} TTLs. Ready for PSTH.")

    def run_psth(self):
        if self.df_analog is None or self.corrected_selected is None or not self.onsets:
            messagebox.showwarning("PSTH", "Run 'Fit & TTL detection' first."); return
        time = self.df_analog[self.cb_time.get().strip()].values.astype(float)
        sig = np.asarray(self.corrected_selected, dtype=float)
        pre = float(self.pre_s.get()); post = float(self.post_s.get())
        sr = auto_sampling_rate(time)

        aligned, rel = interp_align(sig, time, self.onsets, pre, post, sampling_rate=sr)
        if aligned.size == 0:
            messagebox.showwarning("PSTH", "No trials after alignment (check pre/post)."); return
        # Baseline frames where rel < 0 ([-pre, 0))
        baseline_frames = np.where(rel < 0)[0]
        if baseline_frames.size == 0:
            # fallback to first 10% frames
            baseline_frames = np.arange(max(1, int(0.1 * aligned.shape[1])))

        # dF/F using mean baseline
        aligned_dff = dff_from_baseline_mean(aligned, baseline_frames)

        # Optional z-score
        if self.use_zscore.get():
            final = zscore_by_baseline(aligned_dff, baseline_frames)
            ylab = "Z-score (dF/F)"
        else:
            final = aligned_dff
            ylab = "dF/F"

        self.aligned_mat = final
        self.rel_grid = rel

        # Plot mean ± SEM
        self.ax_mean.clear()
        mean = final.mean(axis=0); sem = final.std(axis=0) / max(1, int(np.sqrt(final.shape[0])))
        self.ax_mean.plot(rel, mean, linewidth=1.5)
        self.ax_mean.fill_between(rel, mean - sem, mean + sem, alpha=0.25)
        self.ax_mean.axvline(0, linestyle="--", linewidth=1)
        self.ax_mean.set_xlabel("Time (s)"); self.ax_mean.set_ylabel(ylab)
        self.ax_mean.set_title(f"Mean ± SEM ({final.shape[0]} trials)")
        self.ax_mean.set_xlim(rel[0], rel[-1])

        # Heatmap
        self.ax_heat.clear()
        self.ax_heat.imshow(final, aspect="auto", origin="lower", extent=[rel[0], rel[-1], 0, final.shape[0]])
        self.ax_heat.set_xlabel("Time (s)"); self.ax_heat.set_ylabel("Trial")
        self.ax_heat.set_title("Aligned heatmap")
        self.ax_heat.set_xlim(rel[0], rel[-1])
        self.canvas_psth.draw()

    def save_aligned(self):
        if self.aligned_mat is None:
            messagebox.showwarning("Save", "Run PSTH first."); return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")], initialfile="aligned_dff.csv")
        if not path:
            return
        df = pd.DataFrame(self.aligned_mat, columns=[f"{t:.4f}" for t in self.rel_grid])
        df.to_csv(path, index=False)
        messagebox.showinfo("Saved", f"Saved: {os.path.basename(path)}")

if __name__ == "__main__":
    app = FPApp()
    app.mainloop()