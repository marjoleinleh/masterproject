import sys
from matplotlib.colors import LogNorm
from helper import *
import csv
import statistics
import math
import numpy as np
from tqdm import tqdm

"""
Program that looks at data from the ET scope
"""
init_plotting_preferences()
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
	toalist = np.zeros(len(voltages))
	toalist[:] = np.nan

	tstep = times[10][11] - times[10][10]

	#Filtering bad events
	print("filtering bad events")
	for event in range(len(voltages)):
		peak = True
		if np.amax(voltages[event]) == np.nan:
			deviatingevents.append(event)
			#break
		
		#make lists of the maximum heights per event	
		fheights.append(np.max(voltages[event]))

		#Calculating the standard deviations and means of the first third of the 
		baseline = np.mean(voltages[event][0:int(0.33*len(voltages[event]))])
		stdev = np.std(voltages[event][0:int(0.33*len(voltages[event]))])
		fmeans.append(baseline)
		fstdvs.append(stdev)

		#Check maximum value of the last 20% of the data, removing events that have a late peak
		latemax = np.max(voltages[event][int(0.8*len(voltages[event])):len(voltages[event])])
		latebaseline = np.mean(voltages[event][int(0.8*len(voltages[event])):len(voltages[event])])
		latestdev = np.std(voltages[event][int(0.8*len(voltages[event])):len(voltages[event])])

		#Check if the standard deviation in the last 20% of the data is 3* larger than the sigma of the first third of the data:
		if latestdev > 3*stdev:
			deviatingevents.append(event)
			voltages[event] = len(voltages[event])*[np.nan]
			peak = False
			#break

		earlymax = np.max(voltages[event][0:int(0.33*len(voltages[event]))])
		#print(latemax)
		#if latemax >  baseline + 3*stdev:
			#deviatingevents.append(event)
			#voltages[event] = len(voltages[event])*[np.nan]
			#peak = False
		#elif earlymax >  baseline + 3*stdev:
			#deviatingevents.append(event)
			#voltages[event] = len(voltages[event])*[np.nan]
			#peak = False
		if peak == True:
			if fheights[-1] < baseline + 5 *stdev:
				deviatingevents.append(event)
				voltages[event] = len(voltages[event])*[np.nan]
				peak = False
				#break
				#print("found no peak")
		
		midpoint =int(0.5*len(voltages[event]))

		if peak ==True:
			for datapoint in range(midpoint-20,midpoint):
				if voltages[event][datapoint]< baseline -3*stdev:
					deviatingevents.append(event)
					voltages[event] = len(voltages[event])*[np.nan]
					peak = False
					break
				elif voltages[event][datapoint]< -0.0025:
					deviatingevents.append(event)
					voltages[event] = len(voltages[event])*[np.nan]
					peak = False
					break

		area = 0
		
		if peak == True:
			#Calculating the integrated area
			
			"""
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
			# 110 indices should be integrated
			"""
			for datapoint in range(len(voltages[event])):
				if voltages[event][datapoint]>baseline +5*stdev:
					#if datapoint < 250:
						#deviatingevents.append(event)
						#voltages[event] = len(voltages[event])*[np.nan]
						#peak = False
						#print("found a rogue peak")
					#print(event, datapoint)
					#elif datapoint > 550:
						#deviatingevents.append(event)
						#voltages[event] = len(voltages[event])*[np.nan]
						#peak = False
						#print("found a rogue peak")						
					#else:
					ToA = datapoint *tstep
					toalist[event] = ToA
					#print("STarted integrating at index: ", datapoint)
					for i in range(110):
						area += voltages[event][datapoint +i]
					break
				#Filter signals that have a very negative signal before the 350th datapoint
				if voltages[event][datapoint]<baseline -5*stdev:
					if datapoint < 350:
						deviatingevents.append(event)
						voltages[event] = len(voltages[event])*[np.nan]
						peak = False
						#print("found a low early signal")
						break
						
		if area == 0:
			#deviatingevents.append(event)
			#voltages[event] = len(voltages[event])*[np.nan]
			peak = False
			areas.append(np.nan)
			#print("found no peak")
		#print(event, "integrated signal,", area)	

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
		elif area !=0:
			areas.append(area)


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

	
	#for event in range(len(fheights)):
	#	if fheights[event] == max(fheights):
	# 			maxevents.append(event)
	#	else:
	#		nonmaxheights.append(fheights[event])
	#		nonmaxareas.append(areas[event])
	#print("number of events that were maxed out ",len(maxevents))

	#Figure of all waveforms from fast output
	plt.figure()
	for event in range(len(voltages)):#int(len(voltages2))):
		plt.plot(range(len(voltages[event])), voltages[event], color='black', alpha =0.05)
	plt.xlabel("Time(s)")
	plt.ylabel("Voltage(V)")
	plt.title(savename + "V signals output channel " + channelnum)
	plt.savefig(savename +"V_channel" + channelnum+"_overlay.png")
	#plt.show()
	
	print("Analyse events in one channel")
	return fheights, areas, toalist

def main(filename, ppe, savename):
	times, voltages1, voltages2 = readfile(filename, ppe)
	# test_file(timespart, voltages1, voltages2)
	# sys.exit(0)
	# make_waveforms(voltages, times)

	peaks1, areas1, toalist1 = analyse_channel(times, voltages1, savename, "1")
	peaks2, areas2, toalist2 = analyse_channel(times, voltages2, savename, "2")

	fheights = peaks1 +peaks2
	areas = areas1+areas2

	#sys.exit(0)

	timedifferences = []
	heightdifferences = []
	#peak heights and time difference
	for event in range(len(voltages1)):
		if voltages1[event][0] == np.nan:
			print("skip")
		else:
			peak1 = np.max(voltages1[event])
			peak2 = np.max(voltages2[event])
			#print(peak1,peak2)
			#print(times[event])
			#print(peak1, peak2)
			timediff = toalist1[event] - toalist2[event]
			timedifferences.append(timediff)
			heightdifferences.append(peak1-peak2)

	#print(heightdifferences)
	#print(timedifferences)

	
	return fheights, areas, heightdifferences, timedifferences, peaks1, peaks2

print("Analysis of 26V")
fheights26, areas26, heightdifferences26, timedifferences26, peaks1_26, peaks2_26 = main(r"C:\Users\User\Documents\GRAPPA\masterproject\Short_rod_coincidence\26V-4cm-1000events.csv", 640, "26")
print("Analysis of 27V")
fheights27, areas27, heightdifferences27, timedifferences27, peaks1_27, peaks2_27 = main(r"C:\Users\User\Documents\GRAPPA\masterproject\Short_rod_coincidence\27v-2.5mv_newtrigger.csv", 640, "27")
print("Analysis of 28V")
fheights28, areas28, heightdifferences28, timedifferences28, peaks1_28, peaks2_28 = main(r"C:\Users\User\Documents\GRAPPA\masterproject\Short_rod_coincidence\28V-4cm-2mv.csv", 640, "28")
print("Analysis of 29V")
fheights29, areas29, heightdifferences29, timedifferences29, peaks1_29, peaks2_29 = main(r"C:\Users\User\Documents\GRAPPA\masterproject\Short_rod_coincidence\29v-5.81mvtr.csv", 640, "29")
plt.show()
#sys.exit(0)
plt.figure()
plt.hist2d(peaks1_29, peaks2_29, bins = (np.linspace(-0.005,0.03,50),np.linspace(-0.005,0.03,50)), cmin =1)
plt.xlabel('Peak height SiPM1 (V)')
plt.ylabel('Peak height SiPM1 (V)')
plt.title('Peak heights in each SiPM (29V)')
plt.colorbar()
plt.savefig('peakheight_correlation_29.png')
#plt.show()

plt.figure()
plt.hist2d(peaks1_28, peaks2_28, bins = (np.linspace(-0.005,0.03,50),np.linspace(-0.005,0.03,50)), cmin =1)
plt.xlabel('Peak height SiPM1 (V)')
plt.ylabel('Peak height SiPM1 (V)')
plt.title('Peak heights in each SiPM (28V)')
plt.colorbar()
plt.savefig('peakheight_correlation_28V.png')
#plt.show()

plt.figure()
plt.hist2d(peaks1_27, peaks2_27, bins = (np.linspace(-0.005,0.03,50),np.linspace(-0.005,0.03,50)), cmin =1)
plt.xlabel('Peak height SiPM1 (V)')
plt.ylabel('Peak height SiPM1 (V)')
plt.title('Peak heights in each SiPM (27V)')
plt.colorbar()
plt.savefig('peakheight_correlation_27V.png')
#plt.show()

plt.figure()
plt.hist2d(peaks1_26, peaks2_26, bins = (np.linspace(-0.005,0.03,50),np.linspace(-0.005,0.03,50)), cmin =1)
plt.xlabel('Peak height SiPM1 (V)')
plt.ylabel('Peak height SiPM1 (V)')
plt.title('Peak heights in each SiPM (26V)')
plt.colorbar()
plt.savefig('peakheight_correlation_26V.png')
#plt.show()

#histogram of peak heights of fast output
plt.figure()
plt.hist(areas26, bins =np.linspace(-0.2,1,100),histtype = u'step', label = "26 V", density =True)
plt.hist(areas27, bins =np.linspace(-0.2,1,100),histtype = u'step', label = "27 V", density = True)
plt.hist(areas28, bins =np.linspace(-0.2,1,100),histtype = u'step', label = "28 V", density = True)
plt.hist(areas29, bins =np.linspace(-0.2,1,100),histtype = u'step', label = "29 V", density =True)
#plt.hist(areas27, bins = np.linspace(-2, 14, 35), histtype = u'step')
#plt.hist(areas28, bins = np.linspace(-2, 14, 35), histtype = u'step')
#plt.hist(areas29, bins = np.linspace(-2, 14, 35), histtype = u'step')
plt.title("Integrated signals")
plt.xlabel("Signal Area (V)")
plt.ylabel("Counts")
plt.legend()
plt.savefig("integrated_area_all.png")
#plt.show()

plt.figure()

plt.hist(timedifferences26, range =(-0.000000004,0.000000004), bins =50,histtype = u'step', label = "26 V", density =True)
plt.hist(timedifferences27, range =(-0.000000004,0.000000004), bins =50,histtype = u'step', label = "27 V", density =True)
plt.hist(timedifferences28, range =(-0.000000004,0.000000004), bins =50,histtype = u'step', label = "28 V", density =True)
plt.hist(timedifferences29, range =(-0.000000004,0.000000004), bins =50,histtype = u'step', label = "29 V", density =True)
plt.legend()
plt.xlabel('Time difference (s)')
plt.ylabel('Counts')
plt.title('Time difference spectrum')
plt.savefig('Time_Difference_Spectrum.png')
#plt.show()

plt.figure()
plt.hist2d(heightdifferences26, timedifferences26, bins = (np.linspace(-0.02,0.02,30),np.linspace(-0.0000000025,0.0000000025,30)), cmin =1)
plt.xlabel('Peak height differences (V)')
plt.ylabel('Time differences (s)')
plt.title('Peak height and time difference correlation (26V)')
plt.colorbar()
plt.savefig('peak_time_correlation_26V.png')
#plt.show()

plt.figure()
plt.hist2d(heightdifferences27, timedifferences27, bins = (np.linspace(-0.02,0.02,30),np.linspace(-0.0000000025,0.0000000025,30)), cmin =1)
plt.xlabel('Peak height differences (V)')
plt.ylabel('Time differences (s)')
plt.title('Peak height and time difference correlation (27V)')
plt.colorbar()
plt.savefig('peak_time_correlation_27V.png')
#plt.show()

plt.figure()
plt.hist2d(heightdifferences28, timedifferences28, bins = (np.linspace(-0.02,0.02,30),np.linspace(-0.0000000025,0.0000000025,30)), cmin =1)
plt.xlabel('Peak height differences (V)')
plt.ylabel('Time differences (s)')
plt.title('Peak height and time difference correlation (28V)')
plt.colorbar()
plt.savefig('peak_time_correlation_28V.png')
#plt.show()

plt.figure()
plt.hist2d(heightdifferences29, timedifferences29, bins = (np.linspace(-0.02,0.02,30),np.linspace(-0.0000000025,0.0000000025,30)), cmin =1)
plt.xlabel('Peak height differences (V)')
plt.ylabel('Time differences (s)')
plt.title('Peak height and time difference correlation (29V)')
plt.colorbar()
plt.savefig('peak_time_correlation_29V.png')
#plt.show()

plt.figure()
plt.hist2d(fheights26, areas26, bins = (np.linspace(0,0.04,25),np.linspace(-0.2,1,25)), cmin =1)
plt.xlabel('Peak heights (V)')
plt.ylabel('Integrated signal (V)')
plt.title('Peak Height and Area Correlation (26V)')
plt.colorbar()
plt.savefig('correlation_26V.png')
#plt.show()

"""
plt.figure()
plt.hist2d(nonmaxheights26, nonmaxareas26, bins = (np.linspace(0,0.06,25),np.linspace(-0.2,1,25)), cmap=plt.cm.jet, cmin =1)
plt.xlabel('Peak heights (V)')
plt.ylabel('Integrated signal (V)')
plt.title('Peak Height and Area Correlation (26V)')
plt.colorbar()
plt.savefig('nonmaxcorrelation_26V.png')
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
"""

plt.figure()
plt.hist2d(fheights27, areas27, bins = (np.linspace(0,0.04,25),np.linspace(-0.2,1,25)), cmin =1)
plt.xlabel('Peak heights (V)')
plt.ylabel('Integrated signal (V)')
plt.title('Peak Height and Area Correlation (27V)')
plt.colorbar()
plt.savefig('correlation_27V.png')
#plt.show()

plt.figure()
plt.hist2d(fheights28, areas28, bins = (np.linspace(0,0.04,25),np.linspace(-0.2,1,25)), cmin = 1)
plt.xlabel('Peak heights (V)')
plt.ylabel('Integrated signal (V)')
plt.title('Peak Height and Area Correlation (28V)')
plt.colorbar()
plt.savefig('correlation_28V.png')
#plt.show()

plt.figure()
plt.hist2d(fheights29, areas29, bins = (np.linspace(0,0.04,25),np.linspace(-0.2,1,25)), cmin = 1)
plt.xlabel('Peak heights (V)')
plt.ylabel('Integrated signal (V)')
plt.title('Peak Height and Area Correlation (29V)')
plt.colorbar()
plt.savefig('correlation_29V.png')
#plt.show()



plt.figure()

plt.hist(fheights26, bins = np.linspace(0,0.04,100),histtype = u'step', label = "26 V", density =True)
plt.hist(fheights27, bins = np.linspace(0,0.04,100),histtype = u'step', label = "27 V", density =True)
plt.hist(fheights28, bins = np.linspace(0,0.04,100),histtype = u'step', label = "28 V", density =True)
plt.hist(fheights29, bins = np.linspace(0,0.04,100),histtype = u'step', label = "29 V", density =True)
#plt.yscale('log')
plt.title("Peak heights of Fast Output")
plt.xlabel("Peak height")
plt.ylabel("Counts")
plt.legend()
plt.savefig("peak_heights_all.png")
#plt.show()

#plt.show()

