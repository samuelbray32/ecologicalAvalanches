README

--Files for analysis and plotting of results for the article:
	Forecasting...TODO
	Samuel R. Bray, Bo Wang
	2019

--Files and necessary functions are self-contained in each folder unless otherwise noted
--Files written for cell-by-cell running and visualization


___________________________________________________________________________________________

GENERAL ORGANIZATION:

carbonFlux, mussel, plankton: Segmenting and analysis of Avalanche Events for Fig.1, S1 
	and Table S1-S2

multNoise: Simulation of models (Eq.1). Segmenting and analysis of Avalanche Events for Fig. 1-2, S1 
	and Table S1-S2

combinedPlots: Plotting for Fig 1. Regression Fitting for Table S3.
	*Will Add 10/03/19

Predictability: Forecasting analysis for Fig. 3-4.

___________________________________________________________________________________________

FOLDER: multNoise/

	linResponse.py: Models Eq.1, Segments and saves Avalanches
	multAnalysis_AIC_T.py: Analysis for Table S1
	multAnalysis_AIC_S.py: Analysis for Table S2

FOLDER: multNoise/FigS1/
	Analysis of threshold dependency for Fig. S1

___________________________________________________________________________________________

FOLDERS: plankton/, mussel/, carbonFlux/,
	
	***_analysis.py: loads data, segments avalanches
	***_AIC_T.py: Analysis for Table S1
	***_AIC_S.py: Analysis for Table S2

	plankton/FigS1/diffThresholds.py: Analysis for Fig S1 A,C
	

___________________________________________________________________________________________

FOLDER: Predictability/

	smartPredict_plusBarnacles.py: Runs predictions on some species, including all mollusk groups

	smartPredict_ALLPLANKTON.py: Runs predictions on all plankton species

FOLDER: Predictability/Flux/

	predictability_Carbon: Runs Predictions on HA1 dataset

FOLDER: Predictability/allResults_fig4/
	
	Generates Plot for Fig4
	***NOTE: need to manually move pickle files in if you re-run prediction algorith in other
		folders

FOLDER: Predictability/multNoise/
	Benchmarking of Predictions on linResponse Model

	fig3c_benchmark.py: runs predictions for Fig 3c.
	fig3d_benchmark.py: runs predictions for Fig 3d.
	fig3c_plot.py: plots results for Fig 3c.
	fig3d_plot.py: plots results for Fig 3d.

	

