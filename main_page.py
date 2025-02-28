import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import json
import os
import webbrowser
from ai_swing_analysis import AISwingAnalysis

class StockAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Swing Trading Suite")
        self.root.geometry("1000x800")
        self.root.configure(bg="#F5F6F5")

        self.swing_analyzer = AISwingAnalysis()

        self.load_watchlist()

        style = ttk.Style()
        style.configure("TLabel", background="#F5F6F5", font=("Helvetica", 10))
        style.configure("TButton", font=("Helvetica", 10))
        style.configure("TFrame", background="#F5F6F5")
        style.configure("Treeview", font=("Helvetica", 10), rowheight=25)
        style.configure("Treeview.Heading", font=("Helvetica", 10, "bold"))

        self.watchlist_frame = ttk.LabelFrame(self.root, text="Swing Watchlist", style="TFrame")
        self.watchlist_frame.pack(pady=5, padx=5, fill="x")
        ttk.Label(self.watchlist_frame, text="Tip: Add tickers to track short-term trading opportunities.", font=("Helvetica", 8, "italic")).pack()

        self.swing_tree = ttk.Treeview(self.watchlist_frame, columns=("Ticker", "Company", "Price", "Day Range", "52-Wk Range", "Signal"), show="headings", height=10)
        self.swing_tree.heading("Ticker", text="Ticker")
        self.swing_tree.heading("Company", text="Company Name")
        self.swing_tree.heading("Price", text="Current Price")
        self.swing_tree.heading("Day Range", text="Day Range")
        self.swing_tree.heading("52-Wk Range", text="52-Wk Range")
        self.swing_tree.heading("Signal", text="Signal")
        self.swing_tree.column("Ticker", width=80, anchor="center")
        self.swing_tree.column("Company", width=250)
        self.swing_tree.column("Price", width=100, anchor="center")
        self.swing_tree.column("Day Range", width=150, anchor="center")
        self.swing_tree.column("52-Wk Range", width=150, anchor="center")
        self.swing_tree.column("Signal", width=100, anchor="center")
        self.swing_tree.pack(fill="x")
        self.update_swing_tree()

        button_frame = ttk.Frame(self.watchlist_frame, style="TFrame")
        button_frame.pack(pady=5)
        ttk.Button(button_frame, text="Add Ticker", command=self.add_swing_ticker).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Remove Ticker", command=self.remove_swing_ticker).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Update Watchlist", command=self.update_watchlist).pack(side="left", padx=5)

        self.swing_dropdown = ttk.Combobox(self.watchlist_frame, values=[f"{t} - {self.get_company_info(t, 'name')}" for t in self.swing_watchlist], font=("Helvetica", 10))
        self.swing_dropdown.pack(pady=5)
        ttk.Button(self.watchlist_frame, text="Analyse Selected", command=self.analyze_swing_ticker).pack(pady=5)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_watchlist(self):
        try:
            with open("watchlist.json", "r") as f:
                data = json.load(f)
                self.swing_watchlist = data.get("swing", ["AAPL", "TSLA"])
        except (FileNotFoundError, json.JSONDecodeError):
            self.swing_watchlist = ["AAPL", "TSLA"]

    def save_watchlist(self):
        try:
            data = {"swing": self.swing_watchlist}
            with open("watchlist.json", "w") as f:
                json.dump(data, f, indent=4)
        except PermissionError:
            messagebox.showerror("Save Error", "Cannot save changes—check file permissions or close other programs using watchlist.json.")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save watchlist: {str(e)}")

    def on_closing(self):
        self.save_watchlist()
        self.root.destroy()

    def update_swing_tree(self):
        for item in self.swing_tree.get_children():
            self.swing_tree.delete(item)
        for ticker in self.swing_watchlist:
            info = self.get_company_info(ticker, force_update=True)
            self.swing_tree.insert("", "end", values=(ticker, info['name'], f"{info['currency']}{info['current_price']:.2f}", 
                                                     info['day_range'], info['week_52_range'], info['signal']))

    def update_watchlist(self):
        """Refresh watchlist with latest data."""
        try:
            self.update_swing_tree()
            messagebox.showinfo("Update Successful", "Watchlist updated with the latest information.\nTip: Check for fresh prices and signals.")
        except Exception as e:
            messagebox.showerror("Update Error", f"Failed to update watchlist: {str(e)}")

    def add_swing_ticker(self):
        ticker = simpledialog.askstring("Add Ticker", "Enter ticker symbol (e.g., AAPL):").upper()
        if ticker and ticker not in self.swing_watchlist:
            self.swing_watchlist.append(ticker)
            self.update_swing_tree()
            self.swing_dropdown['values'] = [f"{t} - {self.get_company_info(t, 'name')}" for t in self.swing_watchlist]

    def remove_swing_ticker(self):
        selection = self.swing_tree.selection()
        if selection:
            ticker = self.swing_tree.item(selection[0])['values'][0]
            self.swing_watchlist.remove(ticker)
            self.update_swing_tree()
            self.swing_dropdown['values'] = [f"{t} - {self.get_company_info(t, 'name')}" for t in self.swing_watchlist]

    def get_company_info(self, ticker, key=None, force_update=False):
        try:
            analysis = self.swing_analyzer.analyze_ticker(ticker)
            info = {
                'name': analysis['company_name'],
                'current_price': analysis['current_price'],
                'day_range': analysis['day_range'],
                'week_52_range': analysis['week_52_range'],
                'signal': analysis['signal'],
                'currency': analysis['currency']
            }
            return info if key is None else info[key]
        except Exception:
            return {'name': ticker, 'current_price': 0, 'day_range': 'N/A', 'week_52_range': 'N/A', 'signal': 'N/A', 'currency': '$'}[key] if key else {'name': ticker, 'current_price': 0, 'day_range': 'N/A', 'week_52_range': 'N/A', 'signal': 'N/A', 'currency': '$'}

    def analyze_swing_ticker(self):
        ticker = self.swing_dropdown.get().split(" - ")[0]
        if ticker:
            result = self.swing_analyzer.analyze_ticker(ticker)
            messagebox.showinfo("Swing Analysis", f"{result['summary']}\n\nDetailed Report:\n{result['report']}")
            webbrowser.open(result['visualization'])

if __name__ == "__main__":
    root = tk.Tk()
    app = StockAnalysisApp(root)
    root.mainloop()