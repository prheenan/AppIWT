// Use modern global access method, strict compilation
#pragma rtGlobals=3	

#include "::InverseWeierstrass"
#include ":::UtilIgorPro:Util:IoUtil"
#include ":::UtilIgorPro:Util:PlotUtil"
#pragma ModuleName = ModMainIWT

Static StrConstant DEF_PATH_NAME = "ExampleIWT"

Static Function Main_Windows()
	// Runs a simple IWT on patrick's windows setup
	ModMainIWT#Main()
End Function 

Static Function Main_Mac()
	// Runs a simple IWT on patrick's mac setup 
	ModMainIWT#Main()
End Function

Static Function Main([example,base])
	// // This function shows how to use the IWT code
	// Args:
	//		example: which example to run (0: unfolding, 1: refolding, 2: both)
	//		base: the folder where the Research Git repository lives 
	Variable example
	String base
	if (ParamIsDefault(base))
		base = ModIoUtil#pwd_igor_path(DEF_PATH_NAME,n_up_relative=3)
	EndIf
	if (ParamIsDefault(example))
		example = 2
	EndIf
	String input_file = base
	if (example == 0)
		input_file += "Data/JustUnfold.pxp"	
	elseif (example == 1)
		input_file += "Data/JustRefold.pxp"	
	else
		input_file += "Data/UnfoldandRefold.pxp"
	endif
	KillWaves /A/Z
	ModPlotUtil#clf()
	// IWT options
	Struct InverseWeierstrassOptions opt
	opt.number_of_pairs = 50
	opt.number_of_bins = 80
	opt.z_0 = 0e-9
	opt.velocity_m_per_s = 50e-9
	opt.kbT = 4.1e-21
	opt.f_one_half_N = 12e-12
	opt.flip_forces = 0
	opt.unfold_only = (example == 0)
	opt.refold_only = (example == 1)
	opt.meta.path_to_input_file = input_file
	opt.meta.path_to_research_directory = base
	// Make the output waves
	Struct InverseWeierstrassOutput output
	Make /O/N=0, output.molecular_extension_meters
	Make /O/N=0, output.energy_landscape_joules 
	Make /O/N=0, output.tilted_energy_landscape_joules
	// Execte the command
	ModInverseWeierstrass#inverse_weierstrass(opt,output)
	// Make a fun plot wooo
	ModPlotUtil#figure(hide=0)
	ModPlotUtil#Plot(output.energy_landscape_joules,mX=output.molecular_extension_meters)
	ModPlotUtil#xlabel("Extension (m)")
	ModPlotUtil#ylabel("G (J)")
End Function
