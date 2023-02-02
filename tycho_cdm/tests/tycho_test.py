import os
import unittest

import tycho


class TychoTest(unittest.TestCase):
    def test_dir_is_empty(self):
        here = os.path.abspath(os.path.dirname(__file__))
        new_folder_path = os.path.join(here, 'new_folder')
        os.makedirs(new_folder_path)
        self.assertEqual(tycho.dir_is_empty(new_folder_path), True)
        os.removedirs(new_folder_path)

    def test_dir_is_not_empty(self):
        here = os.path.abspath(os.path.dirname(__file__))
        new_folder_path = os.path.join(here, 'new_folder')
        os.makedirs(new_folder_path)
        new_file_path = os.path.join(new_folder_path, 'new_file.txt')
        with open(new_file_path, 'w'):
            self.assertEqual(tycho.dir_is_empty(new_folder_path), False)
        os.remove(new_file_path)
        os.removedirs(new_folder_path)


if __name__ == '__main__':
    unittest.main()
