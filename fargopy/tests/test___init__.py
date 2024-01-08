import unittest
from fargopy import Dictobj, Fargobj

class TestDictobj(unittest.TestCase):
    def test_initialization(self):
        # Test initialization with keyword arguments
        ob = Dictobj(a=2, b=3)
        self.assertEqual(ob.a, 2)
        self.assertEqual(ob.b, 3)

        # Test initialization with a dictionary
        ob = Dictobj(dict={'a': 2, 'b': 3})
        self.assertEqual(ob.a, 2)
        self.assertEqual(ob.b, 3)

    def test_keys(self):
        ob = Dictobj(a=2, b=3)
        self.assertEqual(ob.keys(), ['a', 'b'])

    def test_item(self):
        ob = Dictobj(a=2, b=3)
        self.assertEqual(ob.item('a'), 2)
        self.assertEqual(ob.item('b'), 3)

        # Test accessing item using __getitem__
        self.assertEqual(ob['a'], 2)
        self.assertEqual(ob['b'], 3)

class TestFargobj(unittest.TestCase):
    def test_initialization(self):
        # Test initialization with keyword arguments
        ob = Fargobj(a=2, b=3)
        self.assertEqual(ob.a, 2)
        self.assertEqual(ob.b, 3)

        # Test initialization with a dictionary
        ob = Fargobj(dict={'a': 2, 'b': 3})
        self.assertEqual(ob.a, 2)
        self.assertEqual(ob.b, 3)

    def test_save_object(self):
        ob = Fargobj(a=2, b=3)
        ob.save_object(filename='/tmp/test_fargobj.json', verbose=True)
        # Add assertions to check if the file was saved correctly

    def test_set_property(self):
        ob = Fargobj(a=2, b=3)
        ob.set_property('c', 4)
        self.assertEqual(ob.c, 4)
        ob.set_property('d', 5, method=lambda prop: prop * 2)
        self.assertEqual(ob.d, 10)

    def test_has(self):
        ob = Fargobj(a=2, b=3)
        self.assertTrue(ob.has('a'))
        self.assertTrue(ob.has('b'))
        self.assertFalse(ob.has('c'))

if __name__ == '__main__':
    unittest.main()