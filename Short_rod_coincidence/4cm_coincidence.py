import sys
from matplotlib.colors import LogNorm
import csv
import matplotlib.pyplot as plt
import statistics
import math
import numpy as np
from tqdm import tqdm

"""
Program that looks at data from the ET scope
"""

#ppe = 200 #points per event

def readfile(filename, ppe):
	"""
	Funtion that reads the data from a sipm and returns lists of lists of the times and voltages for each event
	"""

	with open(filename) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',', )

		#print(csv_reader[2])
		#Matrix made from the number of events and number of data points      
		
		voltages1 = np.zeros((1000, ppe),dtype=np.float64)
		voltages2 = np.zeros((1000,ppe),dtype=np.float64)
		times = np.zeros((1000, ppe),dtype=np.float64)
		line_count = 0#

		#Skips header
		next(csv_reader)
		next(csv_reader)
		next(csv_reader)
		#
		i = 0 #Datapoint within event indix
		event = 0 #Event number
		with tqdm(total = ppe*1000) as progress_bar:
			for row in tqdm(csv_reader):
				times[event][i] = float(row[0])
				voltages1[event][i] = float(row[1])
				voltages2[event][i] = float(row[2])
				i+=1
				
				if i == ppe:
					event+=1
					i=0
				progress_bar.update(1) 

		print(line_count)
		#print("number of events " ,len(voltages1))
		#print(voltages1[0], voltages1[-1])
	return times, voltages1, voltages2

def make_waveforms(voltages, times):
	#plot of a single waveform, both output modes
	for event in range(3):
		plt.figure()
		plt.scatter(times[event], voltages[event], label = 'Fast Output')#, alpha = 0.5)	
		print(np.min(times[event]), np.max(times[event]))
		#plt.plot(times[event], [baseline[event]]*(len(times[event])), label = "Baseline" )
		plt.legend()
		plt.title('Waveform G1, event '+str(event))
		plt.xlabel('Time (s)')
		plt.ylabel('Voltage (V)')
		#plt.savefig(str(event)+'waveform.png')
		plt.show()

def test_file(times, voltage1, voltage2):
	make_waveforms(voltage1, times)
	make_waveforms(voltage2, times)
	plt.figure()
	plt.hist2d(voltage1.flatten(), voltage2.flatten(), bins = 250, norm = LogNorm())
	plt.show()

	plt.hist(times.flatten(), bins = 250)
	plt.show()

	for time in times:
		assert len(time) == 640 #Checks that there are 640 times in one event
	
	assert np.isfinite(times.all())
	assert np.isfinite(voltage1.all())
	assert np.isfinite(voltage2.all())

def analyse_channel(times, voltages, savename, channelnum):
	fheights = []
	fmeans = []
	fstdvs = []
	areas = []
	deviatingevents = []

	#Filtering bad events
	print("filtering bad events")
	for event in range(len(voltages)):
		peak = False
		if np.amax(voltages[event]) == np.nan:
			deviatingevents.append(event)
			break
		
		#make lists of the maximum heights per event	
		fheights.append(np.max(voltages[event]))

		#Calculating the standard deviations and means of the first third of the 
		baseline = np.mean(voltages[event][0:int(0.33*len(voltages[event]))])
		stdev = np.std(voltages[event][0:int(0.33*len(voltages[event]))])
		fmeans.append(baseline)
		fstdvs.append(stdev)

		#Check maximum value of the last 20% of the data, removing events that have a late peak
		latemax = np.max(voltages[event][int(0.8*len(voltages[event])):len(voltages[event])])
		#print(latemax)
		if latemax > baseline + 1.5*stdev:
			deviatingevents.append(event)
			voltages[event] = len(voltages[event])*[np.nan]

		#Calculating the integrated area
		area = 0
		if savename == "28":
			#print(len(voltages[event]))
			if len(voltages[event])>425:
				for datapoint in np.arange(315, 425):
					area += voltages[event][datapoint]
			elif len(voltages[event])<425:
				print(event, "Event range too small")
				#print(datapoint, voltages[event][datapoint], event)
		elif savename == "27":
			for datapoint in range(295, 405):
				area += voltages[event][datapoint]

		elif savename == "26":
			for datapoint in range(295, 405):
				area += voltages[event][datapoint]
			"""
			if datapoint < fmeans[-1]-7*fstdvs[-1]:
				if event not in deviatingevents:
					deviatingevents.append(event)
					voltages[event] = len(voltages[event])*[np.nan]
				#voltages3[event] = len(voltages2[event])*[np.nan]
				break
			"""
			#elif datapoint > fmeans[-1]+5*fstdvs[-1]:
				#peak = True
			#if datapoint > 0.001:
				#peak = True
				#break
		#print(integrated)
		
		#if peak == False:
			#if event not in deviatingevents:
				#deviatingevents.append(event)
				#voltages[event] = len(voltages[event])*[np.nan]
				#voltages3[event] = len(voltages2[event])*[np.nan]
		areas.append(area)

	"""	
	for event in deviatingevents:
		voltages[event] = len(voltages[event])*[np.nan]
		times[event] = len(voltages[event])*[np.nan]

		if event < len(voltages1):
			voltages1[event] = len(voltages1[event])*[np.nan]
			times1[event] = len(voltages1[event])*[np.nan]
		else:
			voltages2[event-len(voltages1)] = len(voltages2[event-len(voltages1)])*[np.nan]
			times2[event-len(voltages1)] = len(voltages2[event-len(voltages1)])*[np.nan]
	"""

	#print(deviatingevents)
	wrong_events = len(deviatingevents)/len(voltages)
	print("The fraction of events that were dismissed is ",wrong_events)
	
	#optional calling to make the waveform plots
	#make_waveforms(voltages, times, fmeans, fstdvs, baselinelist)

	# plt.figure()
	# plt.hist2d(heightssipm1, heightssipm2, cmap=plt.cm.jet, cmin =1)
	# plt.xlabel('SiPM 1')
	# plt.ylabel('SiPM 2')
	# plt.title('Peak height correlation' +savename +" V")
	# plt.colorbar()

	# plt.savefig(savename + "signal_height.png" )
	# plt.show()

	#checking for cut off peaks
	maxevents = []
	nonmaxheights = []
	nonmaxareas = []

	
	for event in range(len(fheights)):
		if fheights[event] == max(fheights):
			maxevents.append(event)
		else:
			nonmaxheights.append(fheights[event])
			nonmaxareas.append(areas[event])
	print("number of events that were maxed out ",len(maxevents))

	#Figure of all waveforms from fast output
	plt.figure()
	for event in range(len(voltages)):#int(len(voltages2))):
		plt.plot(range(len(voltages[event])), voltages[event], color='black', alpha =0.05)
	plt.xlabel("Time(s)")
	plt.ylabel("Voltage(V)")
	plt.title(savename + "V signals output channel " + channelnum)
	plt.savefig(savename +"V_channel" + channelnum+"_overlay.png")
	plt.show()
	
	print("Analyse events in one channel")
	return fheights, areas

def main(filename, ppe, savename):
	times, voltages1, voltages2 = readfile(filename, ppe)
	# test_file(timespart, voltages1, voltages2)
	# sys.exit(0)
	# make_waveforms(voltages, times)

	peaks1, areas1 = analyse_channel(times, voltages1, savename, "1")
	peaks2, areas2 = analyse_channel(times, voltages2, savename, "2")

	sys.exit(0)

	timedifferences = []
	heightdifferences = []
	#peak heights and time difference
	for event in range(len(voltages1)):
		if voltages1[event][0] == np.nan:
			print("skip")
		else:
			peak1 = max(voltages1[event])
			peak2 = max(voltages2[event])
			#print(peak1, peak2)
			timediff = times1[event][voltages1[event].index(peak1)] - times2[event][voltages2[event].index(peak2)]
			timedifferences.append(timediff)
			heightdifferences.append(peak1-peak2)

	#print(heightdifferences)
	#print(timedifferences)

	
	return fheights, areas, nonmaxheights, nonmaxareas, heightdifferences, timedifferences


fheights28, areas28, nonmaxheights28, nonmaxareas28, heightdifferences28, timedifferences28 = main(r"C:\Users\User\Documents\GRAPPA\masterproject\Short_rod_coincidence\28V-4cm-2mv.csv", 640, "28")
fheights26, areas26, nonmaxheights26, nonmaxareas26, heightdifferences26, timedifferences26 = main(r"C:\Users\User\Documents\GRAPPA\masterproject\Short_rod_coincidence\26V-4cm-1000events.csv", 640, "26")
fheights27, areas27, nonmaxheights27, nonmaxareas27, heightdifferences27, timedifferences27 = main(r"C:\Users\User\Documents\GRAPPA\masterproject\Short_rod_coincidence\27V-4cm-1000events.csv", 640, "27")

plt.figure()
plt.hist(timedifferences26, range =(-0.000000004,0.000000004), bins =25,histtype = u'step', label = "26 V")
plt.hist(timedifferences27, range =(-0.000000004,0.000000004), bins =25,histtype = u'step', label = "27 V")
plt.hist(timedifferences28, range =(-0.000000004,0.000000004), bins =25,histtype = u'step', label = "28 V")
plt.legend()
plt.xlabel('Time difference (s)')
plt.ylabel('Counts')
plt.title('Time difference spectrum')
plt.savefig('Time_Difference_Spectrum.png')
plt.show()

plt.figure()
plt.hist2d(heightdifferences26, timedifferences26, bins = (np.linspace(-0.02,0.02,30),np.linspace(-0.0000000025,0.0000000025,30)), cmap=plt.cm.jet, cmin =1)
plt.xlabel('Peak height differences (V)')
plt.ylabel('Time differences (s)')
plt.title('Peak height and time difference correlation (26V)')
plt.colorbar()
plt.savefig('peak_time_correlation_26V.png')
plt.show()

plt.figure()
plt.hist2d(heightdifferences27, timedifferences27, bins = (np.linspace(-0.02,0.02,30),np.linspace(-0.0000000025,0.0000000025,30)), cmap=plt.cm.jet, cmin =1)
plt.xlabel('Peak height differences (V)')
plt.ylabel('Time differences (s)')
plt.title('Peak height and time difference correlation (27V)')
plt.colorbar()
plt.savefig('peak_time_correlation_27V.png')
plt.show()

plt.figure()
plt.hist2d(heightdifferences28, timedifferences28, bins = (np.linspace(-0.02,0.02,30),np.linspace(-0.0000000025,0.0000000025,30)), cmap=plt.cm.jet, cmin =1)
plt.xlabel('Peak height differences (V)')
plt.ylabel('Time differences (s)')
plt.title('Peak height and time difference correlation (28V)')
plt.colorbar()
plt.savefig('peak_time_correlation_28V.png')
plt.show()

plt.figure()
plt.hist2d(fheights26, areas26, bins = (np.linspace(0,0.04,25),np.linspace(-0.2,0.5,25)), cmap=plt.cm.jet, cmin =1)
plt.xlabel('Peak heights (V)')
plt.ylabel('Integrated signal (V)')
plt.title('Peak Height and Area Correlation (26V)')
plt.colorbar()
plt.savefig('correlation_26V.png')
plt.show()

plt.figure()
plt.hist2d(nonmaxheights26, nonmaxareas26, bins = (np.linspace(0,0.06,25),np.linspace(-0.2,1,25)), cmap=plt.cm.jet, cmin =1)
plt.xlabel('Peak heights (V)')
plt.ylabel('Integrated signal (V)')
plt.title('Peak Height and Area Correlation (26V)')
plt.colorbar()
plt.savefig('nonmaxcorrelation_26V.png')
plt.show()

plt.figure()
plt.hist2d(fheights27, areas27, bins = (np.linspace(0,0.04,25),np.linspace(-0.2,0.5,25)), cmap=plt.cm.jet, cmin =1)
plt.xlabel('Peak heights (V)')
plt.ylabel('Integrated signal (V)')
plt.title('Peak Height and Area Correlation (27V)')
plt.colorbar()
plt.savefig('correlation_27V.png')
plt.show()

plt.figure()
plt.hist2d(fheights28, areas28, bins = (np.linspace(0,0.04,25),np.linspace(-0.2,0.5,25)), cmap=plt.cm.jet, cmin = 1)
plt.xlabel('Peak heights (V)')
plt.ylabel('Integrated signal (V)')
plt.title('Peak Height and Area Correlation (28V)')
plt.colorbar()
plt.savefig('correlation_28V.png')
plt.show()

plt.figure()
plt.hist2d(nonmaxheights28, nonmaxareas28, bins = (np.linspace(0,0.06,25),np.linspace(-0.2,1,25)), cmap=plt.cm.jet, cmin = 1)
plt.xlabel('Peak heights (V)')
plt.ylabel('Integrated signal (V)')
plt.title('Peak Height and Area Correlation (28V)')
plt.colorbar()
plt.savefig('non_max_correlation_28V.png')
plt.show()



plt.figure()
plt.hist(fheights26, bins = np.linspace(0,0.04,50),histtype = u'step', label = "26 V")
plt.hist(fheights27, bins = np.linspace(0,0.04,50),histtype = u'step', label = "27 V")
plt.hist(fheights28, bins = np.linspace(0,0.04,50),histtype = u'step', label = "28 V")
#plt.yscale('log')
plt.title("Peak heights of Fast Output")
plt.xlabel("Peak height")
plt.ylabel("Counts")
plt.legend()
plt.savefig("peak_heights_all.png")
plt.show()

plt.figure()
plt.hist(nonmaxheights26, bins = np.linspace(0,0.04,50),histtype = u'step', label = "26 V")
plt.hist(nonmaxheights27, bins = np.linspace(0,0.04,50),histtype = u'step', label = "27 V")
plt.hist(nonmaxheights28, bins = np.linspace(0,0.04,50),histtype = u'step', label = "28 V")
#plt.yscale('log')
plt.title("Peak heights of Fast Output")
plt.xlabel("Peak height")
plt.ylabel("Counts")
plt.legend()
plt.savefig("nonmaxpeak_heights_all.png")
plt.show()

#histogram of peak heights of fast output
plt.figure()
plt.hist(areas26, bins =np.linspace(-0.2,0.5,50),histtype = u'step', label = "26 V")
plt.hist(areas27, bins =np.linspace(-0.2,0.5,50),histtype = u'step', label = "27 V")
plt.hist(areas28, bins =np.linspace(-0.2,0.5,50),histtype = u'step', label = "28 V")
#plt.yscale('log')
#plt.hist(areas27, bins = np.linspace(-2, 14, 35), histtype = u'step')
#plt.hist(areas28, bins = np.linspace(-2, 14, 35), histtype = u'step')
#plt.hist(areas29, bins = np.linspace(-2, 14, 35), histtype = u'step')
plt.title("Integrated signals")
plt.xlabel("Signal Area (V)")
plt.ylabel("Counts")
plt.legend()
plt.savefig("integrated_area_all.png")
plt.show()

#histogram of peak heights of fast output
plt.figure()
plt.hist(nonmaxareas26, bins =np.linspace(-0.2,0.5,50),histtype = u'step', label = "26 V")
plt.hist(nonmaxareas27, bins =np.linspace(-0.2,0.5,50),histtype = u'step', label = "27 V")
plt.hist(nonmaxareas28, bins =np.linspace(-0.2,0.5,50),histtype = u'step', label = "28 V")
#plt.yscale('log')
#plt.hist(areas27, bins = np.linspace(-2, 14, 35), histtype = u'step')
#plt.hist(areas28, bins = np.linspace(-2, 14, 35), histtype = u'step')
#plt.hist(areas29, bins = np.linspace(-2, 14, 35), histtype = u'step')
plt.title("Integrated signals")
plt.xlabel("Signal Area (V)")
plt.ylabel("Counts")
plt.legend()
plt.savefig("nonmaxintegrated_area_all.png")
plt.show()