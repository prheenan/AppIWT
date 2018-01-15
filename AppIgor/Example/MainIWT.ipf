// Use modern global access method, strict compilation
#pragma rtGlobals=3	

#include "::InverseWeierstrass"
#pragma ModuleName = ModMainIWT

Static StrConstant DEF_INPUT_REL_TO_BASE =  "Applications/AppIWT/AppIgor/Example/input.pxp"

Static Function Main_Windows()
	// Runs a simple IWT on patrick's windows setup
	ModMainIWT#Main("C:/Users/pahe3165/src_prh/")
End Function 

Static Function Main_Mac()
	// Runs a simple IWT on patrick's mac setup 
	ModMainIWT#Main("/Users/patrickheenan/src_prh/")
End Function

Static Function Main(base,[input_file])
	// // This function shows how to use the IWT code
	// Args:
	//		base: the folder where the Research Git repository lives 
	//		input_file: the pxp to load. If not present, defaults to 
	//		<base>DEF_INPUT_REL_TO_BASE
	String base,input_file
	if (ParamIsDefault(input_file))
		input_file  = base +DEF_INPUT_REL_TO_BASE
	EndIf
	KillWaves /A/Z
	// IWT options
	Struct InverseWeierstrassOptions opt
	opt.number_of_pairs = 16
	opt.number_of_bins = 150
	opt.z_0 = 20e-9
	opt.velocity_m_per_s = 20e-9
	opt.kbT = 4.1e-21
	opt.f_one_half_N = 8e-12
	opt.flip_forces = 0
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
	Display output.tilted_energy_landscape_joules vs output.molecular_extension_meters
End Function
