# use sionna functions to simulate intercell interference in NR set-up
import pickle
import argparse
import numpy as np
import os
# turn off the device INFO messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from sionna.channel.tr38901 import AntennaArray, UMi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid
from sionna.channel import gen_single_sector_topology_interferers, subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel, set_3gpp_scenario_parameters


parser = argparse.ArgumentParser()
parser.add_argument('--num_ut_ant', type=int, default=1, help='Number of antennas at the UT')
parser.add_argument('--num_bs_ant', type=int, default=2, help='Number of antennas at the BS')
parser.add_argument('--num_ut', type=int, default=5, help='Number of UTs')
parser.add_argument('--num_interferer', type=int, default=4, help='Number of potential interfering UTs')
parser.add_argument('--num_scheduled_interferer', type=int, default=3, help='Number of scheduled interfering UTs')
parser.add_argument('--batch_size', type=int, default=20, help='Batch size')

args = parser.parse_args()
num_ofdm_symbols = 14
num_streams_per_tx = args.num_ut_ant
rx_tx_association = np.array([[1]])

# set seeds
seed = 10
tf.random.set_seed(seed)
np.random.seed(seed)

# Create antenna arrays
# !!! The number of antennas is dependent on the polarization;
# if dual, then the number of antennas is doubled; if single, then the number of antennas is not changed
bs_array = AntennaArray(num_rows = int(args.num_bs_ant/2),
                        num_cols= 1,
                        polarization = 'dual',
                        polarization_type = 'VH',
                        antenna_pattern = '38.901',
                        carrier_frequency = 3.5e9)

ut_array = AntennaArray(num_rows = 1,
                      num_cols = 2,
                      polarization = 'single',
                      polarization_type = 'V',
                      antenna_pattern = 'omni',
                      carrier_frequency = 3.5e9)
print(bs_array.num_ant.numpy())
print(ut_array.num_ant.numpy())

# Create channel model
channel_model = UMi(carrier_frequency = 3.5e9,
                    o2i_model = 'low',
                    ut_array = ut_array,
                    bs_array = bs_array,
                    direction = 'uplink')

# Generate the topology
topology = gen_single_sector_topology_interferers(batch_size = args.batch_size,
                                                  num_ut = args.num_ut,
                                                  num_interferer = args.num_interferer,
                                                  scenario = 'umi')
# retrun a batch of topologies 
# a single BS located at the origin, num_ut UTs uniformly drooped in a cell sector, 
# and num_interferer interfering UTs dropped in the adjacent cell sectors

# Set the topology
ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state = topology
channel_model.set_topology(ut_loc,
                           bs_loc,
                           ut_orientations,
                           bs_orientations,
                           ut_velocities,
                           in_state)


# ---------------------------------------------------------------------------- #
# Check the Umi parameters
# Set valid parameters for a specified 3GPP system level scenario (RMa, UMi, or UMa).

min_bs_ut_dist, isd,bs_height, min_ut_height, max_ut_height, indoor_probability, min_ut_velocity, max_ut_velocity = set_3gpp_scenario_parameters('umi')
print("Minimum BS-UT distance: ", min_bs_ut_dist.numpy())
print("Inter-site distance: ", isd.numpy())
print("BS elevation: ", bs_height.numpy())
print("Minimum UT elevation: ", min_ut_height.numpy())
print("Maximum UT elevation: ", max_ut_height.numpy())
print("Probability of a UT to be indoor: ", indoor_probability.numpy())
print("Minimum UT velocity: ", min_ut_velocity.numpy())
print("Maximim UT velocity: ", max_ut_velocity.numpy())

# ---------------------------------------------------------------------------- #
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

sample_topo_id = 1

# Plot BS location
ax.scatter(bs_loc[sample_topo_id, 0, 0], bs_loc[sample_topo_id, 0, 1], bs_loc[sample_topo_id, 0, 2], c='b', marker='^', label='BS')
# Plot UT locations
num_ut = args.num_ut
num_interferer = args.num_interferer
num_total_ut = num_ut + num_interferer
ax.scatter(ut_loc[sample_topo_id, 0:num_ut, 0], 
           ut_loc[sample_topo_id, 0:num_ut, 1], 
           ut_loc[sample_topo_id, 0:num_ut, 2], c='r', marker='o', label='UT')
ax.scatter(ut_loc[sample_topo_id, num_ut:num_total_ut, 0],
           ut_loc[sample_topo_id, num_ut:num_total_ut, 1], 
           ut_loc[sample_topo_id, num_ut:num_total_ut, 2], c='g', marker='o', label='UT-interferer')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.legend()
plt.savefig('sample_topology_{:d}interfUT_{:d}totalUT.png'.format(num_interferer, num_total_ut))

# ---------------------------------------------------------------------------- #
sm = StreamManagement(rx_tx_association, num_streams_per_tx)
rg = ResourceGrid(num_ofdm_symbols=14,
                  fft_size=76,
                  subcarrier_spacing=15e3,
                  num_tx=1,
                  num_streams_per_tx=num_streams_per_tx,
                  cyclic_prefix_length=6,
                  num_guard_carriers=[5,6],
                  dc_null=True,
                  pilot_pattern="kronecker",
                  pilot_ofdm_symbol_indices=[2,11])
# apply_channel_freq = ApplyOFDMChannel(add_awgn=True)

cir = channel_model(num_ofdm_symbols, 1/rg.ofdm_symbol_duration)
frequencies = subcarrier_frequencies(num_subcarriers = 76, subcarrier_spacing=15e3)
batch_h_freq = cir_to_ofdm_channel(frequencies, *cir, normalize=True) # this is real channel freq response


# ---------------------------------------------------------------------------- #
# Average over the resource grid, use tensor operation, use MRC optimal receive beamforming (max-SNR, matched filter)
# ID 0~num_ut: serve UTs
# ID num_ut~(num_ut + num_interferer): interfering UTs

# the dimensions of batch_h_freq
# – Channel frequency responses at frequencies
# ([batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size], tf.complex) 
interf_idx = np.random.choice(range(num_ut, num_ut + num_interferer), args.num_scheduled_interferer, replace=False)

SIR_db_list = []
for topo_id in range(args.batch_size):
    for serve_id in range(num_ut):
        ps = 0.
        pi = 0.
        serve_ut_h = batch_h_freq[topo_id, 0,:,serve_id,0,:,:] # [num_bs_ant, num_time_steps, fft_size]
        interf_ut_h_list = [batch_h_freq[topo_id, 0,:,i,0,:,:] for i in interf_idx]

        comm_channel_gain = tf.reduce_sum(tf.abs(serve_ut_h)**2) 
        # print("Communication channel gain: {:.2f}".format(comm_channel_gain))

        # MRC optimal receive beamforming with perfect CSI
        rx_beamformer = tf.math.conj(serve_ut_h) 
        flatten_rx_bf = tf.reshape(rx_beamformer, [-1])
        flatten_serve_ut_h = tf.reshape(serve_ut_h, [-1])
        ps = tf.abs(tf.tensordot(flatten_rx_bf, flatten_serve_ut_h, axes = 1))**2/ comm_channel_gain

        for interf_ut_h in interf_ut_h_list:
            interf_channel_gain = tf.reduce_sum(tf.abs(interf_ut_h)**2)
            flatten_interf_ut_h = tf.reshape(interf_ut_h, [-1])
            pi += tf.abs(tf.tensordot(flatten_rx_bf, flatten_interf_ut_h, axes = 1))**2/ comm_channel_gain
            # print("Interfering channel gain: {:.2f}".format(interf_channel_gain))
        
        ps_over_pi = (ps/pi).numpy()
        ps_over_pi_db = 10 * np.log10(ps_over_pi)
        # print("SIR in dB: {:.2f}".format(ps_over_pi_db))

        SIR_db_list.append(ps_over_pi_db)

print("Average SIR in dB: {:.2f}".format(np.mean(SIR_db_list)))
# ---------------------------------------------------------------------------- #