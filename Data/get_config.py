# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import sionna


def get_online_config():
    '''
    For online data generation
    '''
    channel_config_list = [] # totally 24 configs
    model_type_list = ['TDL']
    PDP_mode_list = ['B', 'C', 'E']
    delay_spread_list = [50e-9, 200e-9, 500e-9] # short, nominal, long, very long (this should be in unit 's'...)
    delta_delay_spread = 20e-9
    mobility_level = [(0.0,5.0), (15.0,20.0)] # mobility level: low, mid, high

    # ---------------------------------------------------------------------------- #
    for model_type in model_type_list:
        for PDP_mode in PDP_mode_list:
            for delay_spread in delay_spread_list:
                for min_speed, max_speed in mobility_level:
                    config = dict()
                    config['model_type'] = model_type
                    config['PDP_mode'] = PDP_mode
                    config['delay_spread'] = delay_spread
                    config['min_speed'] = min_speed
                    config['max_speed'] = max_speed
                    config['delta_delay_spread'] = delta_delay_spread
                    channel_config_list.append(config)

    other_config = dict()
    other_config['num_bs_ant'] = 2
    other_config['fft_size'] = 72
    other_config['num_bits_per_symbol'] = 4
    other_config['pilot_ofdm_symbol_indices'] = [2,11]
    # other_config['low_SIR'] = {'min_SIR': 0, 'max_SIR': 5, 'num_SIR_points': 6} # high interference
    # other_config['mid_SIR'] = {'min_SIR': 5, 'max_SIR': 10, 'num_SIR_points': 6} # medium interference
    other_config['high_SIR'] = {'min_SIR': 10, 'max_SIR': 15, 'num_SIR_points': 6} # low interference

    # limit online data snr range
    other_config['snr_range'] = {'min_ebNo': 3, 'max_ebNo': 10, 'num_ebNo_points': 8} 
    
    return channel_config_list, other_config


def get_offline_config():
    '''
    For offline data generation
    '''
    channel_config_list = [] # totally 48 configs
    model_type_list = ['TDL']
    PDP_mode_list = ['A']
    delay_spread_list = [50e-9] # short, nominal, long, very long (this should be in unit 's'...)
    delta_delay_spread = 20e-9
    mobility_level = [(0,1.5)] # mobility level: low, mid, high

    for model_type in model_type_list:
        for PDP_mode in PDP_mode_list:
            for delay_spread in delay_spread_list:
                for min_speed, max_speed in mobility_level:
                    config = dict()
                    config['model_type'] = model_type
                    config['PDP_mode'] = PDP_mode
                    config['delay_spread'] = delay_spread
                    config['min_speed'] = min_speed
                    config['max_speed'] = max_speed
                    config['delta_delay_spread'] = delta_delay_spread
                    channel_config_list.append(config)

    other_config = dict()
    other_config['num_bs_ant'] = 2
    other_config['fft_size'] = 72
    other_config['num_bits_per_symbol'] = 4 # 16QAM
    other_config['pilot_ofdm_symbol_indices'] = [2,11]
    # other_config['low_SIR'] = {'min_SIR': 0, 'max_SIR': 5, 'num_SIR_points': 6} # high interference
    # other_config['mid_SIR'] = {'min_SIR': 5, 'max_SIR': 10, 'num_SIR_points': 6} # medium interference
    other_config['high_SIR'] = {'min_SIR': 10, 'max_SIR': 15, 'num_SIR_points': 6} # low interference
    other_config['snr_range'] = {'min_ebNo': -5, 'max_ebNo': 14, 'num_ebNo_points': 20}
    return channel_config_list, other_config


def get_cka_eval_config():
    '''
    For CKA dataset comparison
    '''
    channel_config_list = [] # totally 48 configs
    model_type_list = ['TDL']
    PDP_mode_list = ['C']
    delay_spread_list = [50e-9] # short, nominal, long, very long (this should be in unit 's'...)
    delta_delay_spread = 20e-9
    mobility_level = [(0,1.5)] # mobility level: low, mid, high

    for model_type in model_type_list:
        for PDP_mode in PDP_mode_list:
            for delay_spread in delay_spread_list:
                for min_speed, max_speed in mobility_level:
                    config = dict()
                    config['model_type'] = model_type
                    config['PDP_mode'] = PDP_mode
                    config['delay_spread'] = delay_spread
                    config['min_speed'] = min_speed
                    config['max_speed'] = max_speed
                    config['delta_delay_spread'] = delta_delay_spread
                    channel_config_list.append(config)

    other_config = dict()
    other_config['num_bs_ant'] = 2
    other_config['fft_size'] = 72
    other_config['num_bits_per_symbol'] = 4 # 16QAM
    other_config['pilot_ofdm_symbol_indices'] = [2,11]
    other_config['low_SIR'] = {'min_SIR': 0, 'max_SIR': 5, 'num_SIR_points': 6} # high interference
    other_config['mid_SIR'] = {'min_SIR': 5, 'max_SIR': 10, 'num_SIR_points': 6} # medium interference
    other_config['high_SIR'] = {'min_SIR': 10, 'max_SIR': 15, 'num_SIR_points': 6} # low interference
    other_config['snr_range'] = {'min_ebNo': -4, 'max_ebNo': 5, 'num_ebNo_points': 10}
    return channel_config_list, other_config


class BasicConfig():
    # study the impact of the number of BS antennas, fft_size, the number of bits per symbol, and the number of pilots
    def __init__(self,
                 num_bs_ant: int = 2,
                 fft_size: int = 72,
                 num_bits_per_symbol: int = 4,
                 pilot_ofdm_symbol_indices: list = [2,11]
                 ):
        super().__init__()

        # Default channel PDP files
        self._PDP_list = ['A', 'B', 'C', 'D', 'E']
        
        self._cyclic_prefix_length = 16
        self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices # two pilot configuration: the 2nd and 11th OFDM symbols are pilots
        self._num_ut_ant = 1
        self._num_bs_ant = num_bs_ant
        self._carrier_frequency = 4e9
        self._subcarrier_spacing = 15e3
        # self._fft_size = 276
        self._fft_size = fft_size # number of subcarriers = 48 = 4 PRBs
        self._num_ofdm_symbols = 14 # per slot
        self._num_streams_per_tx = 1
        self._dc_null = True
        self._num_guard_carriers = [5, 6]
        self._pilot_pattern = "kronecker"
        self._num_bits_per_symbol = num_bits_per_symbol # 16QAM
        self._coderate = 658/1024


        # Required system components
        self._sm = sionna.mimo.StreamManagement(np.array([[1]]),
                                                self._num_streams_per_tx)
        
        self._rg = sionna.ofdm.ResourceGrid(num_ofdm_symbols=self._num_ofdm_symbols,
                                            fft_size=self._fft_size,
                                            subcarrier_spacing = self._subcarrier_spacing,
                                            num_tx=1,
                                            num_streams_per_tx=self._num_streams_per_tx,
                                            cyclic_prefix_length=self._cyclic_prefix_length,
                                            num_guard_carriers=self._num_guard_carriers,
                                            dc_null=self._dc_null,
                                            pilot_pattern=self._pilot_pattern,
                                            pilot_ofdm_symbol_indices=self._pilot_ofdm_symbol_indices)
        
        # all the databits carried by the resource grid with size `fft_size`x`num_ofdm_symbols` form a single codeword.
        self._n = int(self._rg.num_data_symbols * self._num_bits_per_symbol) # Codeword length. 
        self._k = int(self._n*self._coderate) # Number of information bits per codeword

        self._ut_array = sionna.channel.tr38901.Antenna(polarization="single",
                                                        polarization_type="V",
                                                        antenna_pattern="38.901",
                                                        carrier_frequency=self._carrier_frequency)
        if self._num_bs_ant == 1:
            # Single antenna set-up at the BS
            self._bs_array = sionna.channel.tr38901.Antenna(polarization="single",
                                                            polarization_type="V",
                                                            antenna_pattern="38.901",
                                                            carrier_frequency=self._carrier_frequency)
        else:
            # Assume uplink, 1 row, 2 columns, dual polarization, VH --> 4 antenna elements
            self._bs_array = sionna.channel.tr38901.AntennaArray(num_rows=1,
                                                            num_cols= int(self._num_bs_ant/2),
                                                            polarization="dual",
                                                            polarization_type="VH",
                                                            antenna_pattern="38.901",
                                                            carrier_frequency=self._carrier_frequency)


        self._frequencies = sionna.channel.subcarrier_frequencies(self._rg.fft_size, self._rg.subcarrier_spacing)
        
        # Apply channel frequency response
        self._channel_freq = sionna.channel.ApplyOFDMChannel(add_awgn=True)
        self._interf_channel_freq = sionna.channel.ApplyOFDMChannel(add_awgn=False)
        self._binary_source = sionna.utils.BinarySource() # seed = none
        self._encoder = sionna.fec.ldpc.encoding.LDPC5GEncoder(self._k, self._n)
        self._mapper = sionna.mapping.Mapper("qam", self._num_bits_per_symbol)
        self._rg_mapper = sionna.ofdm.ResourceGridMapper(self._rg)
        self._ls_est = sionna.ofdm.LSChannelEstimator(self._rg, interpolation_type="nn") # use the nearest neighbor interpolation
        self._lmmse_equ = sionna.ofdm.LMMSEEqualizer(self._rg, self._sm)
        self._demapper = sionna.mapping.Demapper("app", "qam", self._num_bits_per_symbol) # set hard_out
        self._decoder = sionna.fec.ldpc.decoding.LDPC5GDecoder(self._encoder, hard_out=True)
        self._remove_nulled_scs = sionna.ofdm.RemoveNulledSubcarriers(self._rg)


    def set_channel_models( self,
                            model_type: str= 'TDL',
                            PDP_mode: str = 'A', # choose from ['A', 'B', 'C', 'D', 'E']
                            delay_spread: float = 30e-9,
                            min_speed: float = 0.0, 
                            max_speed: float = 0.0,
                            delta_delay_spread: float = 0.0):
        

        self._PDP = PDP_mode
        self._delay_spread = delay_spread # this should be in the unit of 's'...
        self._min_speed = min_speed
        self._max_speed = max_speed

        # ------------- Generate random channel paramters for interferers ------------ #
        self._model_type = model_type
        self._interf_PDP = np.random.choice(self._PDP_list)
        self._interf_delay_spread = np.random.uniform(low = delay_spread - delta_delay_spread, high = delay_spread + delta_delay_spread)
        self._interf_min_speed = min_speed
        self._interf_max_speed = max_speed # the speed of UT and interferer UT are both uniformly sampled from [min_speed, max_speed]

        # ---------------------------------------------------------------------------- #
        if self._model_type == 'TDL':
            # no spatial correlation is considered
            self._comm_channel_model = sionna.channel.tr38901.TDL(model=self._PDP,
                                                                delay_spread=self._delay_spread,
                                                                carrier_frequency=self._carrier_frequency,
                                                                min_speed=self._min_speed,
                                                                max_speed=self._max_speed,
                                                                num_rx_ant=self._num_bs_ant,
                                                                num_tx_ant=self._num_ut_ant) # set random_seed to generate random uniform doppler, phi, and theta
            
            self._interf_channel_model = sionna.channel.tr38901.TDL(model = self._interf_PDP,
                                                                    delay_spread = self._interf_delay_spread,
                                                                    carrier_frequency = self._carrier_frequency,
                                                                    min_speed = self._interf_min_speed,
                                                                    max_speed = self._interf_max_speed,
                                                                    num_rx_ant = self._num_bs_ant,
                                                                    num_tx_ant = self._num_ut_ant)
        elif model_type == 'CDL':
            self._comm_channel_model = sionna.channel.tr38901.CDL(model=self._PDP,
                                                                delay_spread=self._delay_spread,
                                                                carrier_frequency=self._carrier_frequency,
                                                                ut_array=self._ut_array,
                                                                bs_array=self._bs_array,
                                                                direction="uplink",
                                                                min_speed=self._min_speed,
                                                                max_speed=self._max_speed)
            
            self._interf_channel_model = sionna.channel.tr38901.CDL(model = self._interf_PDP,
                                                                    delay_spread = self._interf_delay_spread,
                                                                    carrier_frequency = self._carrier_frequency,
                                                                    ut_array = self._ut_array,
                                                                    bs_array = self._bs_array,
                                                                    direction = "uplink",
                                                                    min_speed = self._interf_min_speed,
                                                                    max_speed = self._interf_max_speed)
        
        else:
            raise ValueError("model_type must be either TDL or CDL")
