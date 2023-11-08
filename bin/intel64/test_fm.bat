rem .\Release\speech_sample.exe -m .\dparn_c.xml -i .\dparn_input.ark -d CPU -o dparn_c_mkldnn.ark
rem .\Release\speech_sample.exe -m .\dparn_tc.xml -i .\dparn_tc_input.ark -d CPU -o dparn_tc_mkldnn.ark
rem .\Release\speech_sample.exe -m .\mha.xml -i in1=.\mha_in1.ark,in2=.\mha_in2.ark,in3=.\mha_in3.ark -d CPU -o out1/sink_port_0:0=mha_out1_mkldnn.ark,out2/sink_port_0:0=mha_out2_mkldnn.ark

.\Release\speech_sample.exe -m .\dparn_c.xml -i .\dparn_input.ark -d GNA_SW_FP32 -exec_target GNA_TARGET_3_5 -r dparn_c_mkldnn.ark -o dparn_c_gna.ark
.\Release\speech_sample.exe -m .\dparn_c_factorized_pot.xml -i .\dparn_input.ark -d GNA_SW_FP32 -exec_target GNA_TARGET_3_5 -r dparn_c_mkldnn.ark -o dparn_c_pot_gna.ark
.\Release\speech_sample.exe -m .\dparn_c.xml -i .\dparn_input.ark -d GNA_SW_EXACT -exec_target GNA_TARGET_3_5 -r dparn_c_mkldnn.ark -o dparn_c_gna.ark
.\Release\speech_sample.exe -m .\dparn_c_factorized_pot.xml -i .\dparn_input.ark -d GNA_SW_EXACT -exec_target GNA_TARGET_3_5 -r dparn_c_mkldnn.ark -o dparn_c_pot_gna.ark

.\Release\speech_sample.exe -m .\dparn_tc.xml -i .\dparn_tc_input.ark -d GNA_SW_FP32 -exec_target GNA_TARGET_3_5 -r dparn_tc_mkldnn.ark -o dparn_tc_gna.ark
.\Release\speech_sample.exe -m .\dparn_tc_pot.xml -i .\dparn_tc_input.ark -d GNA_SW_FP32 -exec_target GNA_TARGET_3_5 -r dparn_tc_mkldnn.ark -o dparn_tc_pot_gna.ark
.\Release\speech_sample.exe -m .\dparn_tc.xml -i .\dparn_tc_input.ark -d GNA_SW_EXACT -exec_target GNA_TARGET_3_5 -r dparn_tc_mkldnn.ark -o dparn_tc_gna.ark
.\Release\speech_sample.exe -m .\dparn_tc_pot.xml -i .\dparn_tc_input.ark -d GNA_SW_EXACT -exec_target GNA_TARGET_3_5 -r dparn_tc_mkldnn.ark -o dparn_tc_pot_gna.ark

.\Release\speech_sample.exe -m .\mha.xml -i in1=.\mha_in1.ark,in2=.\mha_in2.ark,in3=.\mha_in3.ark -d GNA_SW_FP32 -exec_target GNA_TARGET_3_5 -o out1/sink_port_0:0=mha_out1_gna.ark,out2/sink_port_0:0=mha_out2_gna.ark -r out1/sink_port_0:0=mha_out1_mkldnn.ark,out2/sink_port_0:0=mha_out2_mkldnn.ark -pwl_me 0.025
.\Release\speech_sample.exe -m .\mha_pot.xml -i in1=.\mha_in1.ark,in2=.\mha_in2.ark,in3=.\mha_in3.ark -d GNA_SW_FP32 -exec_target GNA_TARGET_3_5 -o out1/sink_port_0:0=mha_pot_out1_gna.ark,out2/sink_port_0:0=mha_pot_out2_gna.ark -r out1/sink_port_0:0=mha_out1_mkldnn.ark,out2/sink_port_0:0=mha_out2_mkldnn.ark -pwl_me 0.025
.\Release\speech_sample.exe -m .\mha.xml -i in1=.\mha_in1.ark,in2=.\mha_in2.ark,in3=.\mha_in3.ark -d GNA_SW_EXACT -exec_target GNA_TARGET_3_5 -o out1/sink_port_0:0=mha_out1_gna.ark,out2/sink_port_0:0=mha_out2_gna.ark -r out1/sink_port_0:0=mha_out1_mkldnn.ark,out2/sink_port_0:0=mha_out2_mkldnn.ark -pwl_me 0.025
.\Release\speech_sample.exe -m .\mha_pot.xml -i in1=.\mha_in1.ark,in2=.\mha_in2.ark,in3=.\mha_in3.ark -d GNA_SW_EXACT -exec_target GNA_TARGET_3_5 -o out1/sink_port_0:0=mha_pot_out1_gna.ark,out2/sink_port_0:0=mha_pot_out2_gna.ark -r out1/sink_port_0:0=mha_out1_mkldnn.ark,out2/sink_port_0:0=mha_out2_mkldnn.ark -pwl_me 0.025
