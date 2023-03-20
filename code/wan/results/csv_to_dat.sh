# rename all plot_data.csv files to plot_data.dat
for i in `find . -name plot_data.csv`; do
    mv $i `echo $i | sed 's/\.csv/\.dat/'`
done