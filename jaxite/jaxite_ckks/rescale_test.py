"""Tests for Rescale kernel."""

import hypothesis
from hypothesis import strategies as st
import jax
import jax.numpy as jnp
from jaxite.jaxite_ckks import encode
from jaxite.jaxite_ckks import encrypt
from jaxite.jaxite_ckks import key_gen
from jaxite.jaxite_ckks import random
from jaxite.jaxite_ckks import rescale
from jaxite.jaxite_ckks import types
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

jax.config.update("jax_enable_x64", True)


MODULI_1 = [1073742113, 1073740609, 1073741953, 1073741441, 1073741857]
IN_DATA_1 = np.array(
    [
        [
            [423785547, 674572399, 7101169, 717545614, 960621811],
            [867445182, 562016042, 227629453, 537748826, 759889476],
            [314281690, 689811229, 973189237, 732975596, 1065817348],
            [387216809, 645200620, 382116887, 465575012, 937596980],
            [317665009, 890951569, 436205593, 222502279, 332871829],
            [744769517, 589133615, 1049470765, 540842086, 378411034],
            [854727753, 973213608, 618898275, 535809083, 1691508],
            [556289972, 556537284, 211745856, 838630621, 496620154],
            [178254355, 422035805, 424574924, 143568799, 932193751],
            [728938463, 114108925, 346573960, 822209840, 822088981],
            [422880370, 325645582, 420885112, 78316550, 519113977],
            [625974632, 634088838, 908930650, 364841057, 295821080],
            [839454833, 157955718, 155218136, 1059974140, 239696052],
            [265462911, 598501244, 442283445, 272106758, 110480767],
            [794994418, 264734960, 5245353, 437331910, 1044402272],
            [971808730, 731651488, 655703819, 250156030, 425743785],
        ],
        [
            [93050344, 773420993, 869014237, 523961024, 490685917],
            [982589074, 1016391708, 890954083, 59369515, 348525422],
            [696085422, 486533351, 305991144, 805484081, 488819912],
            [903808329, 907193085, 948597079, 142062287, 171056455],
            [1073011810, 276641678, 279786774, 772180452, 578135610],
            [882273633, 795453196, 1060040146, 880685500, 330980879],
            [695828671, 1042614922, 775843178, 36297104, 7463878],
            [583700055, 799111929, 725010039, 477514495, 7389941],
            [717731060, 1069809506, 662779830, 1044291625, 964683342],
            [272965092, 957528200, 173756107, 315124203, 827884344],
            [503830870, 258981671, 797961074, 70786593, 119294455],
            [771893460, 76107404, 171294358, 506398363, 916674206],
            [349583234, 365378699, 119622542, 271859057, 896391895],
            [875122295, 1036703896, 832227271, 901297327, 75091622],
            [468434047, 171269251, 773545209, 248205340, 915263159],
            [317633096, 598535025, 556481132, 1046163582, 812028983],
        ],
    ],
    dtype=np.uint64,
)
OUT_DATA_1 = np.array(
    [
        [
            [124157418, 624907766, 968736221, 920014272],
            [224769381, 506816351, 819405119, 96135897],
            [1020412604, 29862080, 779260731, 606341721],
            [569932777, 728329160, 718396480, 201368550],
            [981325121, 393393842, 902931463, 650843777],
            [470810866, 1072665381, 711592629, 950281688],
            [1029735180, 244219748, 633115307, 969317123],
            [113719130, 661516300, 294975194, 755310602],
            [537113111, 191712908, 367534575, 289763239],
            [970877925, 533435850, 833863491, 26347580],
            [440985463, 216404242, 42215971, 539213324],
            [891573079, 206978512, 630663680, 464126245],
            [282851624, 75342479, 876237112, 20526087],
            [388993781, 179480532, 466398473, 420557824],
            [886080475, 969594190, 208344709, 1065022397],
            [101309031, 598068547, 308403580, 1055579960],
        ],
        [
            [187797345, 286415726, 112913532, 64214583],
            [346468403, 122441288, 1033686805, 953035458],
            [400091616, 468680473, 589830928, 352178779],
            [779213129, 406964333, 753319193, 71249168],
            [237567707, 495076659, 302065675, 817404617],
            [272698807, 202468229, 194084609, 344838098],
            [849445132, 760746605, 587390065, 261594243],
            [901732556, 543463053, 547338915, 796287197],
            [147999934, 669599140, 244128372, 886003104],
            [456593923, 113818155, 495461986, 362926014],
            [481331552, 1048004662, 12041662, 166409376],
            [783750384, 784691089, 957047606, 1062227053],
            [356629236, 34456805, 1057381834, 672133715],
            [281115942, 46178384, 792645433, 579323825],
            [501544985, 176257778, 788806231, 559836544],
            [691247402, 438475456, 537244886, 62465017],
        ],
    ],
    dtype=np.uint64,
)

MODULI_2 = [1073741441, 1073740609, 1073739937]
IN_DATA_2 = np.array(
    [
        [
            [939985, 106511, 387568],
            [285629, 293176, 284224],
            [240895, 718644, 65099],
            [583374, 744105, 844475],
            [304778, 217507, 736975],
            [447533, 207595, 637712],
            [292345, 774037, 387131],
            [620661, 559181, 493496],
            [277695, 108307, 637238],
            [589322, 70227, 747149],
            [290407, 556261, 620850],
            [296197, 688847, 584674],
            [769337, 822056, 230681],
            [320430, 792819, 290000],
            [119570, 769627, 281573],
            [601846, 670383, 207306],
        ],
        [
            [661347, 854792, 838121],
            [625061, 469071, 843356],
            [135321, 259101, 793986],
            [705673, 233965, 346187],
            [676509, 640211, 862332],
            [562079, 198751, 436992],
            [528719, 407261, 390340],
            [890602, 719438, 645712],
            [845416, 816622, 782657],
            [68661, 767320, 811889],
            [120849, 343568, 95466],
            [418908, 756999, 625168],
            [947690, 680003, 346327],
            [336473, 320203, 446308],
            [325277, 437409, 784481],
            [788418, 121648, 692527],
        ],
    ],
    dtype=np.uint32,
)
OUT_DATA_2 = np.array(
    [
        [
            [870864944, 484052509],
            [729210190, 768212470],
            [232648389, 534174526],
            [956253117, 211149183],
            [767851089, 738355584],
            [432594358, 343339028],
            [304223252, 404653314],
            [794533040, 265982529],
            [1047314529, 58454657],
            [963047884, 157929782],
            [374328771, 615668265],
            [280811971, 194404920],
            [873962009, 736002926],
            [104139198, 225634743],
            [641923532, 863288066],
            [733313705, 90401385],
        ],
        [
            [939274298, 751892686],
            [557534218, 619497315],
            [997140158, 12125660],
            [164953145, 922804142],
            [545434074, 272624853],
            [425594591, 136147713],
            [596443480, 691085042],
            [762202801, 13961815],
            [82357464, 532161261],
            [945644448, 586709278],
            [64718163, 411455570],
            [346361451, 143822965],
            [413199545, 623686089],
            [479744085, 462635625],
            [552786021, 1021987441],
            [866468318, 826492234],
        ],
    ],
    dtype=np.uint32,
)


class RescaleTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.degree = 16
    cls.moduli = [1073741441, 1073740609, 1073739937]
    cls.pk, cls.sk = key_gen.keygen(cls.degree, cls.moduli)
    cls.rescale_kernel = rescale.Rescale()
    cls.rescale_kernel.precompute_constants(cls.moduli, 1, 4, 4)
    dummy_ct = types.Ciphertext(
        data=jnp.zeros((2, 16, 3), dtype=jnp.uint32),
        moduli=jnp.array(cls.moduli, dtype=jnp.uint32),
    )
    cls.rescale_kernel.rescale(dummy_ct)

  @parameterized.parameters(
      (MODULI_1, IN_DATA_1, OUT_DATA_1),
      (MODULI_2, IN_DATA_2, OUT_DATA_2),
  )
  def test_rescale_equivalence(self, moduli, in_data, expected_out):
    r, c = 4, 4  # implies degree = 16
    num_rescales = 1

    ct = types.Ciphertext(
        data=jnp.array(in_data), moduli=jnp.array(moduli, dtype=jnp.uint64)
    )

    rescale_kernel = rescale.Rescale()
    rescale_kernel.precompute_constants(moduli, num_rescales, r, c)
    rescale_kernel.rescale(ct)

    np.testing.assert_array_equal(ct.data, expected_out)
    np.testing.assert_array_equal(ct.moduli, moduli[:-1])

  def test_encrypt_decrypt_3_moduli(self):
    scale = 1073739937.0  # match the last modulus

    slots = [complex(1.0, 0)] * 8

    encoder = encode.Encode(self.degree, self.moduli, scale)
    pt = encoder.encode(slots)

    encryptor = encrypt.Encrypt(self.pk)
    random_source = random.TestRandomSource(6789)
    ct = encryptor.encrypt(pt, random_source=random_source)

    decryptor = encrypt.Decrypt(self.sk)
    pt_dec = decryptor.decrypt(ct)

    decoder = encode.Decode(scale, len(slots))
    decoded = decoder.decode(pt_dec)

    for s, d in zip(slots, decoded):
      self.assertAlmostEqual(s.real, d.real, delta=0.5)
      self.assertAlmostEqual(s.imag, d.imag, delta=0.5)

  def test_rescale_e2e(self):
    input_scale = 2**40
    output_scale = 2**40 / self.moduli[-1]

    slots = [complex(1.0, 0)] * 8

    encoder = encode.Encode(self.degree, self.moduli, input_scale)
    pt = encoder.encode(slots)

    encryptor = encrypt.Encrypt(self.pk)
    random_source = random.TestRandomSource(5678)
    ct = encryptor.encrypt(pt, random_source=random_source)

    self.rescale_kernel.rescale(ct)

    decryptor = encrypt.Decrypt(self.sk)
    pt_dec = decryptor.decrypt(ct)

    decoder = encode.Decode(output_scale, len(slots))
    decoded = np.array(decoder.decode(pt_dec))
    expected = np.array(slots)

    np.testing.assert_allclose(decoded, expected, atol=0.5)

  def test_rescale_different_scale(self):
    """Test rescale with a larger scale."""
    input_scale = 2**45
    output_scale = 2**45 / self.moduli[-1]

    slots = [complex(1.0, 0)] * 8

    encoder = encode.Encode(self.degree, self.moduli, input_scale)
    pt = encoder.encode(slots)

    encryptor = encrypt.Encrypt(self.pk)
    random_source = random.TestRandomSource(4567)
    ct = encryptor.encrypt(pt, random_source=random_source)

    self.rescale_kernel.rescale(ct)

    decryptor = encrypt.Decrypt(self.sk)
    pt_dec = decryptor.decrypt(ct)

    decoder = encode.Decode(output_scale, len(slots))
    decoded = np.array(decoder.decode(pt_dec))
    expected = np.array(slots)

    np.testing.assert_allclose(decoded, expected, atol=0.5)

  def test_rescale_more_moduli(self):
    """Test rescale with a larger number of moduli."""
    degree = 16
    moduli = [1073692673, 1073643521, 1073479681, 1073430529]
    input_scale = 2**40
    output_scale = 2**40 / moduli[-1]

    slots = [complex(1.0, 0)] * 8

    encoder = encode.Encode(degree, moduli, input_scale)
    pt = encoder.encode(slots)

    random_source = random.TestRandomSource(3456)
    pk, sk = key_gen.keygen(degree, moduli, random_source=random_source)
    encryptor = encrypt.Encrypt(pk)
    ct = encryptor.encrypt(pt, random_source=random_source)

    rescale_kernel = rescale.Rescale()
    rescale_kernel.precompute_constants(moduli, num_rescales=1, r=4, c=4)
    rescale_kernel.rescale(ct)

    decryptor = encrypt.Decrypt(sk)
    pt_dec = decryptor.decrypt(ct)

    decoder = encode.Decode(output_scale, len(slots))
    decoded = np.array(decoder.decode(pt_dec))
    expected = np.array(slots)

    np.testing.assert_allclose(decoded, expected, atol=0.5)

  def test_double_rescale(self):
    """Test double-rescale."""
    degree = 16
    moduli = [1073692673, 1073643521, 1073479681, 1073430529]

    # We need a larger input scale because double-rescaling
    # will drop by two 30-bit moduli.
    input_scale = 2**63
    output_scale = input_scale / (moduli[-1] * moduli[-2])

    slots = [complex(25.0, 0)] * 8

    encoder = encode.Encode(degree, moduli, input_scale)
    pt = encoder.encode(slots)

    random_source = random.TestRandomSource(2345)
    pk, sk = key_gen.keygen(degree, moduli, random_source=random_source)
    encryptor = encrypt.Encrypt(pk)
    ct = encryptor.encrypt(pt, random_source=random_source)

    rescale_kernel = rescale.Rescale()
    rescale_kernel.precompute_constants(moduli, num_rescales=2, r=4, c=4)
    rescale_kernel.rescale(ct)

    decryptor = encrypt.Decrypt(sk)
    pt_dec = decryptor.decrypt(ct)

    decoder = encode.Decode(output_scale, len(slots))
    decoded = np.array(decoder.decode(pt_dec))
    expected = np.array(slots)

    self.assertLen(ct.moduli, 2)
    np.testing.assert_array_equal(ct.moduli, moduli[:-2])

    np.testing.assert_allclose(decoded, expected, rtol=0.1)

  def test_rescale_different_degree(self):
    """Test rescale with a smaller polynomial degree."""
    degree = 8
    moduli = [1073741441, 1073740609, 1073739937]
    input_scale = 2**40
    output_scale = 2**40 / moduli[-1]

    slots = [complex(1.0, 0)] * 4

    encoder = encode.Encode(degree, moduli, input_scale)
    pt = encoder.encode(slots)

    random_source = random.TestRandomSource(1234)
    pk, sk = key_gen.keygen(degree, moduli, random_source=random_source)
    encryptor = encrypt.Encrypt(pk)
    ct = encryptor.encrypt(pt, random_source=random_source)

    rescale_kernel = rescale.Rescale()
    rescale_kernel.precompute_constants(moduli, 1, 2, 4)
    rescale_kernel.rescale(ct)

    decryptor = encrypt.Decrypt(sk)
    pt_dec = decryptor.decrypt(ct)

    decoder = encode.Decode(output_scale, len(slots))
    decoded = np.array(decoder.decode(pt_dec))
    expected = np.array(slots)

    np.testing.assert_allclose(decoded, expected, atol=0.5)


class RescaleHypothesisTest(absltest.TestCase):

  MODULI = [1073692673, 1073643521, 1073479681]
  DEGREE = 16
  R = 4
  C = 4

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.rescale_kernel = rescale.Rescale()
    cls.rescale_kernel.precompute_constants(cls.MODULI, 1, cls.R, cls.C)
    dummy_ct = types.Ciphertext(
        data=jnp.zeros((2, cls.DEGREE, len(cls.MODULI)), dtype=jnp.uint32),
        moduli=jnp.array(cls.MODULI, dtype=jnp.uint32),
    )
    cls.rescale_kernel.rescale(dummy_ct)
    cls.random_source = random.TestRandomSource(123)
    cls.pk, cls.sk = key_gen.keygen(cls.DEGREE, cls.MODULI, cls.random_source)
    cls.encryptor = encrypt.Encrypt(cls.pk)
    cls.decryptor = encrypt.Decrypt(cls.sk)

  @hypothesis.settings(max_examples=20, deadline=1000)
  @hypothesis.given(
      slots=st.lists(
          st.complex_numbers(min_magnitude=0, max_magnitude=10),
          min_size=8,
          max_size=8,
      ),
      scale_power=st.integers(min_value=40, max_value=50),
  )
  def test_rescale_hypothesis(self, slots, scale_power):
    """Test rescale with randomized inputs using Hypothesis."""
    input_scale = 2**scale_power
    output_scale = input_scale / self.MODULI[-1]

    encoder = encode.Encode(self.DEGREE, self.MODULI, input_scale)
    pt = encoder.encode(slots)

    ct = self.encryptor.encrypt(pt, random_source=self.random_source)
    self.rescale_kernel.rescale(ct)
    pt_dec = self.decryptor.decrypt(ct)

    decoder = encode.Decode(output_scale, len(slots))
    decoded = np.array(decoder.decode(pt_dec))
    expected = np.array(slots)

    # atol=1.0 is used because of randomized inputs up to magnitude 10.
    np.testing.assert_allclose(decoded, expected, atol=1.0)


if __name__ == "__main__":
  absltest.main()
