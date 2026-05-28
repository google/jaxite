"""Tests for equivalence between jaxite_word and jaxite_ckks.

Reference commit from CROSS repo: 69c46d2bf25f017e7f4a24e864ad8abb9506a5c4
"""

import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import encode
from jaxite.jaxite_ckks import encrypt
from jaxite.jaxite_ckks import ntt
from jaxite.jaxite_ckks import ntt_cpu
from jaxite.jaxite_ckks import random
from jaxite.jaxite_ckks import types
import numpy as np
from absl.testing import absltest

jax.config.update('jax_enable_x64', True)


class CrossEquivalenceTest(absltest.TestCase):

  def test_encode_equivalence(self):
    degree = 16
    scale = 563019763943521
    moduli = [1073742881, 1073742721, 1073741441, 1073741857, 524353]

    slots = [
        complex(0.25, 0),
        complex(0.5, 0),
        complex(0.75, 0),
        complex(1, 0),
        complex(2, 0),
        complex(3, 0),
        complex(4, 0),
        complex(5, 0),
    ]

    expected_data = np.array(
        [
            [136867625, 992729062, 1062901950, 64309566, 48023],
            [448217487, 21710597, 452417311, 776229866, 220065],
            [459110530, 1072702960, 135576683, 13425321, 107178],
            [399986586, 819955793, 346580006, 305401790, 87720],
            [338761771, 467831082, 833645419, 435102310, 305645],
            [253416924, 808362513, 770044521, 509826858, 509546],
            [529757713, 1036319305, 429707580, 665276930, 22149],
            [922456207, 726996332, 105156242, 988523858, 10556],
            [56346712, 330072408, 553702770, 645102273, 136241],
            [322124327, 359500508, 1054902493, 1008698515, 420817],
            [822180619, 673061236, 208387012, 877663497, 188715],
            [843740957, 603132359, 321561487, 67265671, 102123],
            [410369751, 388650230, 1058737500, 1009707013, 347437],
            [448727365, 430265802, 497160630, 382861955, 371814],
            [276075424, 22743178, 297873459, 418401481, 384996],
            [309009688, 991696481, 143704361, 422137951, 407445],
        ],
        dtype=np.uint32,
    )

    encoder = encode.Encode()
    encoder.precompute_constants(degree, moduli, scale)
    pt = encoder.encode(slots)

    np.testing.assert_array_equal(np.array(pt.data), expected_data)

  def test_encrypt_equivalence(self):
    degree = 16
    moduli = [1073742881, 1073742721, 1073741441, 1073741857, 524353]

    plaintext_data = np.array(
        [
            [136867625, 992729062, 1062901950, 64309566, 48023],
            [448217487, 21710597, 452417311, 776229866, 220065],
            [459110530, 1072702960, 135576683, 13425321, 107178],
            [399986586, 819955793, 346580006, 305401790, 87720],
            [338761771, 467831082, 833645419, 435102310, 305645],
            [253416924, 808362513, 770044521, 509826858, 509546],
            [529757713, 1036319305, 429707580, 665276930, 22149],
            [922456207, 726996332, 105156242, 988523858, 10556],
            [56346712, 330072408, 553702770, 645102273, 136241],
            [322124327, 359500508, 1054902493, 1008698515, 420817],
            [822180619, 673061236, 208387012, 877663497, 188715],
            [843740957, 603132359, 321561487, 67265671, 102123],
            [410369751, 388650230, 1058737500, 1009707013, 347437],
            [448727365, 430265802, 497160630, 382861955, 371814],
            [276075424, 22743178, 297873459, 418401481, 384996],
            [309009688, 991696481, 143704361, 422137951, 407445],
        ],
        dtype=np.uint64,
    )
    pt = types.Plaintext(
        data=jnp.array(plaintext_data, dtype=jnp.uint32),
        moduli=jnp.array(moduli, dtype=jnp.uint32),
    )

    ones = np.ones((degree, len(moduli)), dtype=np.uint32)
    v_coeffs = ntt_cpu.intt_negacyclic_poly(ones, moduli)
    e_coeffs = ntt_cpu.intt_negacyclic_poly(ones, moduli)

    class MockRandomSource(random.RandomSource):

      def gen_ternary_poly(self, d, m):
        return v_coeffs

      def gen_gaussian_poly(self, d, m, sigma=3.19):
        return e_coeffs

      def gen_uniform_poly(self, d, m):
        return np.zeros((d, len(m)))

    random_source = MockRandomSource()

    pk_data = np.ones((2, degree, len(moduli)), dtype=np.uint32)
    pk = types.PublicKey(data=pk_data, moduli=np.array(moduli, dtype=np.uint32))

    encryptor = encrypt.Encrypt()
    encryptor.precompute_constants(pk)
    ct = encryptor.encrypt(pt, random_source=random_source)

    expected_c0 = np.array(
        [
            [136867627, 992729064, 1062901952, 64309568, 48025],
            [448217489, 21710599, 452417313, 776229868, 220067],
            [459110532, 1072702962, 135576685, 13425323, 107180],
            [399986588, 819955795, 346580008, 305401792, 87722],
            [338761773, 467831084, 833645421, 435102312, 305647],
            [253416926, 808362515, 770044523, 509826860, 509548],
            [529757715, 1036319307, 429707582, 665276932, 22151],
            [922456209, 726996334, 105156244, 988523860, 10558],
            [56346714, 330072410, 553702772, 645102275, 136243],
            [322124329, 359500510, 1054902495, 1008698517, 420819],
            [822180621, 673061238, 208387014, 877663499, 188717],
            [843740959, 603132361, 321561489, 67265673, 102125],
            [410369753, 388650232, 1058737502, 1009707015, 347439],
            [448727367, 430265804, 497160632, 382861957, 371816],
            [276075426, 22743180, 297873461, 418401483, 384998],
            [309009690, 991696483, 143704363, 422137953, 407447],
        ],
        dtype=np.uint32,
    )
    expected_c1 = np.full((degree, len(moduli)), 2, dtype=np.uint32)
    expected_data = np.stack([expected_c0, expected_c1], axis=0)
    np.testing.assert_array_equal(np.array(ct.data), expected_data)

  def test_composition_equivalence(self):
    degree = 16
    scale = 563019763943521
    moduli = [1073742881, 1073742721, 1073741441, 1073741857, 524353]

    slots = [
        complex(0.25, 0),
        complex(0.5, 0),
        complex(0.75, 0),
        complex(1, 0),
        complex(2, 0),
        complex(3, 0),
        complex(4, 0),
        complex(5, 0),
    ]

    encoder = encode.Encode()
    encoder.precompute_constants(degree, moduli, scale)
    pt = encoder.encode(slots)

    class MockRandomSource(random.RandomSource):

      def gen_ternary_poly(self, d, m):
        return np.ones((d, len(m)), dtype=np.uint64)

      def gen_gaussian_poly(self, d, m, sigma=3.19):
        return np.ones((d, len(m)), dtype=np.uint64)

      def gen_uniform_poly(self, d, m):
        return np.zeros((d, len(m)))

    random_source = MockRandomSource()

    pk_data = np.ones((2, degree, len(moduli)), dtype=np.uint64)
    pk = types.PublicKey(data=pk_data, moduli=np.array(moduli, dtype=np.uint64))

    encryptor = encrypt.Encrypt()
    encryptor.precompute_constants(pk)
    ct = encryptor.encrypt(pt, random_source=random_source)

    expected_c0 = np.array(
        [
            [872419520, 8604287, 480934096, 25801204, 330053],
            [469349058, 556713542, 935622343, 1067481183, 45538],
            [715758396, 580337228, 385611780, 813897697, 205816],
            [345567482, 956354630, 966087341, 696048455, 317056],
            [617259511, 893511697, 999916174, 262302662, 219291],
            [587843598, 613946054, 765137346, 1025385380, 237377],
            [328547237, 787114654, 685883158, 359543080, 374572],
            [493244081, 877339909, 1059356805, 538696796, 474001],
            [485558842, 179728835, 673243652, 21187482, 197153],
            [523334807, 608705163, 798726919, 240690512, 68398],
            [487753949, 867477699, 213294191, 362104979, 460888],
            [565243221, 177451748, 155290736, 240065323, 188481],
            [464788859, 252251397, 439230169, 619060352, 118105],
            [192079503, 922631538, 247125537, 656131440, 273180],
            [254943857, 561482958, 888409872, 127150168, 35174],
            [647200678, 902078539, 725672219, 460646317, 125419],
        ],
        dtype=np.uint64,
    )

    expected_c1 = np.array(
        [
            [735551895, 89617946, 491773587, 1035233495, 282030],
            [21131571, 535002945, 483205032, 291251317, 349826],
            [256647866, 581376989, 250035097, 800472376, 98638],
            [1019323777, 136398837, 619507335, 390646665, 229336],
            [278497740, 425680615, 166270755, 900942209, 437999],
            [334426674, 879326262, 1068834266, 515558522, 252184],
            [872532405, 824538070, 256175578, 768008007, 352423],
            [644530755, 150343577, 954200563, 623914795, 463445],
            [429212130, 923399148, 119540882, 449827066, 60912],
            [201210480, 249204655, 817565867, 305733854, 171934],
            [739316211, 194416463, 4907179, 558183339, 272173],
            [795245145, 648062110, 907470690, 172799652, 86358],
            [54419108, 937343888, 454234110, 683095196, 295021],
            [817095019, 492365736, 823706348, 273269485, 425719],
            [1052611314, 538739780, 590536413, 782490544, 174531],
            [338190990, 984124779, 581967858, 38508366, 242327],
        ],
        dtype=np.uint64,
    )

    expected_data = np.stack([expected_c0, expected_c1], axis=0)
    np.testing.assert_array_equal(np.array(ct.data), expected_data)

  def test_ntt_equivalence(self):
    moduli = [2147483489, 2147483137, 2147482817]
    r, c = 4, 4
    degree = r * c
    b = 2

    coef_in_raw = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        [7, 0, 0, 0, 11, 0, 0, 0, 13, 0, 0, 0, 17, 0, 0, 0],
        [16, 1, 15, 2, 14, 3, 13, 4, 12, 5, 11, 6, 10, 7, 9, 8],
    ]
    eval_in_raw = [
        [
            1927271639,
            1976851019,
            961431098,
            750997937,
            1858820200,
            1645119999,
            137255436,
            1444884120,
            379729034,
            612811897,
            1855784060,
            1875694634,
            102353194,
            1048752242,
            1711958033,
            1037636875,
        ],
        [
            1409600302,
            1409600302,
            1409600302,
            1409600302,
            1397968663,
            1397968663,
            1397968663,
            1397968663,
            1703050717,
            1703050717,
            1703050717,
            1703050717,
            1931829757,
            1931829757,
            1931829757,
            1931829757,
        ],
        [
            675784815,
            337807662,
            1571641390,
            400235048,
            1145917821,
            477891965,
            1006973034,
            524384318,
            1690087058,
            137036486,
            1954663574,
            889829120,
            1108726548,
            2133686119,
            1387723090,
            1737474744,
        ],
    ]

    coef_in = jnp.concatenate(
        [
            jnp.array(coef_in_raw, dtype=jnp.uint32)
            .transpose(1, 0)
            .reshape(1, degree, -1)
            for _ in range(b)
        ],
        axis=0,
    )
    eval_in = jnp.concatenate(
        [
            jnp.array(eval_in_raw, dtype=jnp.uint32)
            .transpose(1, 0)
            .reshape(1, degree, -1)
            for _ in range(b)
        ],
        axis=0,
    )

    ntt_kernel = ntt.NTTBarrett()
    ntt_kernel.precompute_constants(moduli, r, c)

    ntt_input = coef_in.reshape(b, r, c, -1)
    ntt_output = ntt_kernel.ntt(ntt_input)

    np.testing.assert_array_equal(eval_in, ntt_output.reshape(b, degree, -1))

    intt_output = ntt_kernel.intt(ntt_output)
    np.testing.assert_array_equal(coef_in, intt_output.reshape(b, degree, -1))


if __name__ == '__main__':
  absltest.main()
