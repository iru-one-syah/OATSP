import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve


def oatsp_phase_kernel(m: float, k: np.ndarray, N: int) -> np.ndarray:
    """
    Compute the phase spectrum of the Optimized Aoshima Time-Stretched Pulse
    (OATSP).

    In the Suzuki et al. paper, the OATSP spectrum is defined using a
    quadratic phase term. This function calculates that phase term for the
    positive-frequency side of the spectrum.

    Parameters
    ----------
    m : float
        Stretch factor of the OATSP.
    k : np.ndarray
        Frequency-bin indices.
    N : int
        Signal length.

    Returns
    -------
    np.ndarray
        Complex phase values of the OATSP spectrum.
    """
    return np.exp(1j * 4 * m * np.pi * (k ** 2) / (N ** 2))


def generate_tsp(i: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate an Optimized Aoshima Time-Stretched Pulse (OATSP) and its inverse
    filter.

    The signal length is N = 2**i. In the original paper, the parameter m
    controls how much the pulse is stretched in time. Here, m is selected from
    a practical stretch map so that the generated TSP can be used for audio
    impulse-response measurement.

    Parameters
    ----------
    i : int
        Exponent for the signal length: N = 2**i.
        The expected range is 9 <= i <= 16.

    Returns
    -------
    tsp : np.ndarray
        Normalized OATSP signal in the time domain.
    inv_tsp : np.ndarray
        Inverse OATSP filter used for deconvolution.
    """
    
    if not 9 <= i <= 16:
        raise ValueError("i must be between 9 and 16 because stretch_map is defined for this range.")

    # ------------------------------------------------------------
    # 1) Define the signal length and stretch factor.
    # ------------------------------------------------------------
    N = 2 ** i

    # According to Fig. 7 in Suzuki et al., choosing the OATSP stretch
    # factor around these values keeps the maximum error below -90 dB.
    # Therefore, the following stretch map is used for practical audio
    # impulse-response measurement.
    stretch_map = np.array([8, 10, 12, 13, 14, 15, 15, 15])
    m = stretch_map[i - 9] * N / 32

    # ------------------------------------------------------------
    # 2) Build the OATSP spectrum.
    # ------------------------------------------------------------
    k = np.arange(N)
    half = N // 2

    H = np.empty(N, dtype=complex)

    # Positive-frequency side, including DC and Nyquist.
    lower_mask = k <= half
    H[lower_mask] = oatsp_phase_kernel(m, k[lower_mask], N)

    # Negative-frequency side.
    # Conjugate symmetry is used so that the inverse FFT becomes a real signal.
    H[~lower_mask] = np.conj(
        oatsp_phase_kernel(m, N - k[~lower_mask], N)
    )

    # ------------------------------------------------------------
    # 3) Convert the spectrum to the time domain.
    # ------------------------------------------------------------
    h = np.real(np.fft.ifft(H))

    # The inverse filter is obtained from the reciprocal spectrum.
    # Convolving the measured response with this inverse filter gives the
    # impulse response.
    G = 1.0 / H
    g = np.real(np.fft.ifft(G)) * np.max(np.abs(h))

    # ------------------------------------------------------------
    # 4) Normalize the TSP signal.
    # ------------------------------------------------------------
    h /= np.max(np.abs(h))

    # ------------------------------------------------------------
    # 5) Circularly rotate the TSP and inverse TSP.
    # ------------------------------------------------------------
    # The paper shows that the OATSP and inverse OATSP are shifted in opposite
    # directions. This rotation places the main part of the signal in a useful
    # position for audio playback and deconvolution.
    shift_h = half - int(m)
    tsp = np.roll(h, -shift_h)

    shift_g = half + int(m)
    inv_tsp = np.roll(g, -shift_g)

    return tsp, inv_tsp


def align_ir(ir: np.ndarray, length: int) -> np.ndarray:
    """
    Center the main impulse-response peak and extract a fixed-length segment.

    This function is only used for plotting. It makes the impulse response
    easier to view by placing the largest peak near the center of the plot.
    """
    peak = np.argmax(np.abs(ir))

    start = peak - length // 2
    start = max(start, 0)

    end = start + length
    if end > len(ir):
        end = len(ir)
        start = max(end - length, 0)

    return ir[start:end]


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    # Generate the OATSP signal and its inverse filter.
    #
    # i = 12 means:
    # N = 2**12 = 4096 samples.
    TSP, INVTSP = generate_tsp(12)

    # Plot the generated OATSP signal.
    plt.figure(figsize=(8, 3))
    plt.plot(TSP)
    plt.title("TSP Signal")
    plt.tight_layout()

    # ------------------------------------------------------------
    # Demo deconvolution → impulse response
    # ------------------------------------------------------------
    # In a real audio measurement, TSP is played through a loudspeaker,
    # earphone, bone-conduction transducer, or other audio device.
    #
    # The recorded signal is then convolved with INVTSP to obtain the
    # impulse response.
    #
    # Here, we convolve TSP with INVTSP only as a simple check.
    # Ideally, this gives a sharp impulse response.
    ir = fftconvolve(TSP, INVTSP)

    # Align the impulse response only for visualization.
    ir_aligned = align_ir(ir, length=len(ir))

    # Plot with the main impulse peak around sample 0.
    x = np.arange(len(ir_aligned)) - len(ir_aligned) / 2

    plt.figure(figsize=(8, 3))
    plt.plot(x, ir_aligned)
    plt.title("Impulse Response")
    plt.tight_layout()

    plt.show()
