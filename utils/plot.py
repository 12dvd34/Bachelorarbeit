import matplotlib.pyplot as plt
# used when plotting test results

data_x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
data_y_l = [2.9, 6.93, 10.7, 14.38, 17.91, 22.31, 26.62, 30.41, 34, 37.19, 39.95, 41.44, 42.57, 43.72, 44.87]
data_y_l2 = [6.43, 15.02, 23.46, 32.44, 41.01, 51.04, 60.34, 67.65, 74.98, 80.54, 85.3, 88.07, 90.5, 92.14, 94.3]
data_y_r = [62, 59.8, 54.1, 43.8]

label_x = "Epoche"
label_y_l = "Präzision (%)"
label_y_r = "Laufzeit (s)"

legend_l = "Testset"
legend_l2 = "Trainingsset"
legend_r = "Laufzeit"

fig, ax_l = plt.subplots()
#ax_r = ax_l.twinx()

ax_l.plot(data_x, data_y_l, color="tab:blue", label=legend_l)
ax_l.plot(data_x, data_y_l2, color="tab:orange", label=legend_l2)
#ax_r.plot(data_x, data_y_r, color="tab:orange", label=legend_r)

ax_l.set_ylabel(label_y_l)
ax_l.set_xlabel(label_x)
#ax_r.set_ylabel(label_y_r)

ax_l.legend(loc="upper left")
#ax_r.legend(loc="upper right")

fig.show()
# this is exactly as dumb as it sounds, but it works
print("set breakpoint here and use debug mode to see figure")
