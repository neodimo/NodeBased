"""Which graphics adapter the app runs on: ranking, the Settings choice, the environment override and the warning."""
import copy
import os
import unittest
from unittest import mock

from nodebased import gpu3d
from nodebased.core import DEFAULT_SETTINGS


class Fake:
    def __init__(self, name, kind, backend='Vulkan', buffer=2 ** 30):
        self.info = dict(device=name, adapter_type=kind, backend_type=backend, description='driver 1')
        self.limits = {'max-buffer-size': buffer, 'max-storage-buffer-binding-size': buffer}


LAPTOP = [Fake('Intel(R) Iris(R) Xe Graphics', 'IntegratedGPU'), Fake('llvmpipe', 'CPU'),
          Fake('NVIDIA GeForce RTX 4070 Laptop GPU', 'DiscreteGPU')]


def summaries(adapters):
    return [gpu3d._summary(a) for a in gpu3d.rank_adapters(adapters)]


class AdapterRanking(unittest.TestCase):
    def test_discrete_beats_integrated_beats_software(self):
        names = [a.info['device'] for a in gpu3d.rank_adapters(LAPTOP)]
        self.assertEqual(names, ['NVIDIA GeForce RTX 4070 Laptop GPU', 'Intel(R) Iris(R) Xe Graphics', 'llvmpipe'])

    def test_a_tie_goes_to_the_adapter_with_more_memory(self):
        small, big = Fake('Small', 'DiscreteGPU', buffer=2 ** 28), Fake('Big', 'DiscreteGPU', buffer=2 ** 33)
        self.assertIs(gpu3d.select_adapter([small, big]), big)

    def test_the_same_card_through_opengl_ranks_after_its_vulkan_entry(self):
        gl, vk = Fake('RTX', 'DiscreteGPU', 'OpenGL', 2 ** 34), Fake('RTX', 'DiscreteGPU', 'Vulkan')
        self.assertIs(gpu3d.select_adapter([gl, vk]), vk)

    def test_nothing_to_choose_from(self):
        self.assertIsNone(gpu3d.select_adapter([]))

    def test_a_named_preference_wins_over_the_ranking(self):
        self.assertEqual(gpu3d.select_adapter(LAPTOP, 'iris').info['device'], 'Intel(R) Iris(R) Xe Graphics')
        self.assertEqual(gpu3d.select_adapter(LAPTOP, 'LLVMPIPE').info['device'], 'llvmpipe')

    def test_automatic_and_an_unknown_name_fall_back_to_the_ranking(self):
        for preference in ('', 'Automatic', 'no such card'):
            self.assertEqual(gpu3d.select_adapter(LAPTOP, preference).info['device'], 'NVIDIA GeForce RTX 4070 Laptop GPU')


class AdapterPreference(unittest.TestCase):
    def test_the_environment_variable_overrides_the_setting(self):
        with mock.patch.dict(os.environ, {gpu3d.ADAPTER_ENV: 'radeon'}):
            self.assertEqual(gpu3d.adapter_preference(), 'radeon')

    def test_the_setting_is_read_when_the_variable_is_unset(self):
        class Store:
            def __init__(self, *a):
                pass

            def value(self, key, default=None):
                return 'Iris' if key == gpu3d.ADAPTER_SETTING else default
        env = {k: v for k, v in os.environ.items() if k != gpu3d.ADAPTER_ENV}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch('PySide6.QtCore.QSettings', Store):
            self.assertEqual(gpu3d.adapter_preference(), 'Iris')


class AdapterWarning(unittest.TestCase):
    def test_warns_on_integrated_when_a_discrete_card_exists(self):
        adapters = summaries(LAPTOP)
        text = gpu3d.adapter_warning('Intel(R) Iris(R) Xe Graphics', adapters)
        self.assertEqual(text, 'Intel(R) Iris(R) Xe Graphics: a discrete adapter is available, Settings → Graphics adapter')
        self.assertIn('llvmpipe: a discrete adapter', gpu3d.adapter_warning('llvmpipe', adapters))

    def test_silent_on_the_discrete_card(self):
        self.assertIsNone(gpu3d.adapter_warning('NVIDIA GeForce RTX 4070 Laptop GPU', summaries(LAPTOP)))

    def test_silent_when_no_discrete_card_exists(self):
        adapters = summaries([Fake('Iris', 'IntegratedGPU'), Fake('llvmpipe', 'CPU')])
        self.assertIsNone(gpu3d.adapter_warning('Iris', adapters))

    def test_silent_for_an_unknown_name(self):
        self.assertIsNone(gpu3d.adapter_warning('', summaries(LAPTOP)))

    def test_the_help_text_names_the_windows_per_app_setting(self):
        self.assertIn('Display', gpu3d.ADAPTER_WARNING_HELP)
        self.assertIn('High performance', gpu3d.ADAPTER_WARNING_HELP)


class TopBarWarning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_the_pill_turns_into_the_warning(self):
        from nodebased.topbar import GpuPill
        pill = GpuPill()
        text = gpu3d.adapter_warning('Iris', summaries([Fake('Iris', 'IntegratedGPU'), Fake('RTX', 'DiscreteGPU')]))
        pill.set_state('Iris', 339, warning=text, help_text=gpu3d.ADAPTER_WARNING_HELP)
        self.assertTrue(pill.warning)
        self.assertIn('Iris: a discrete adapter is available', pill.text())
        self.assertIn('High performance', pill.toolTip())
        pill.set_state('RTX', 12)
        self.assertFalse(pill.warning)
        self.assertEqual(pill.text(), 'RTX · 12 ms')

    def test_settings_lists_automatic_then_each_adapter_once_and_says_restart(self):
        from nodebased.app import ProjectSettingsDialog
        adapters = summaries(LAPTOP) + [dict(summaries(LAPTOP)[0], backend='OpenGL')]
        dialog = ProjectSettingsDialog(copy.deepcopy(DEFAULT_SETTINGS), adapters=adapters, graphics_adapter='llvmpipe')
        combo = dialog.graphics_adapter
        self.assertEqual(combo.itemData(0), '')
        self.assertIn('Automatic', combo.itemText(0))
        self.assertEqual([combo.itemData(i) for i in range(1, combo.count())],
                         ['NVIDIA GeForce RTX 4070 Laptop GPU', 'Intel(R) Iris(R) Xe Graphics', 'llvmpipe'])
        self.assertEqual(combo.currentData(), 'llvmpipe')
        self.assertIn('restart', dialog.graphics_adapter_note.text().lower())


if __name__ == '__main__':
    unittest.main()
