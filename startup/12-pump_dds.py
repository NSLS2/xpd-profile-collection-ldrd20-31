from ophyd import Device, EpicsMotor, EpicsSignal, EpicsSignalRO
from ophyd import Component as Cpt
import time
import pandas as pd
import numpy as np


class syrng_DDS(Device):
    communication = Cpt(EpicsSignal, "}DISABLE", string=True, kind="config")
    # 1: Disabled, 0: Enabled

    update_pump = Cpt(EpicsSignal, "}UPDATE.SCAN", string=True, kind="config")
    # 0: Passive, 1: Event, 2: I/O Intr, 3: 10 second 4: 5 second
    # 5: 2 second, 6: 1 second, 7: .5 second, 8: .2 second, 9: .1 second

    set_condition = Cpt(EpicsSignal, "}CONDITION", string=True, kind="config")
    show_condition = Cpt(EpicsSignalRO, "}CONDITION:RBV", string=True, kind="config")


class syrng_DDS_ax(Device):
    status = Cpt(EpicsSignalRO, "STATUS", string=True)
    # 0: Idle, 1: Infusing, 2: Withdrawing, 4: Target Reached

    pump_infuse = Cpt(EpicsSignal, "IRUN")
    pump_withdraw = Cpt(EpicsSignal, "WRUN")
    pump_stop = Cpt(EpicsSignal, "STOP")

    target_vol = Cpt(EpicsSignal, "TVOLUME", kind="config")
    target_vol_unit = Cpt(EpicsSignal, "TVOLUME:EGU", string=True, kind="config")
    read_target_vol = Cpt(EpicsSignalRO, "TVOLUME:RBV", kind="config")
    read_target_vol_unit = Cpt(
        EpicsSignalRO, "TVOLUME:RBV:EGU", string=True, kind="config"
    )

    clear_infused = Cpt(EpicsSignal, "CIVOLUME")
    clear_withdrawn = Cpt(EpicsSignal, "CWVOLUME")

    read_infused = Cpt(EpicsSignalRO, "IVOLUME:RBV")
    read_withdrawn = Cpt(EpicsSignalRO, "WVOLUME:RBV")
    read_infused_unit = Cpt(EpicsSignalRO, "IVOLUME:RBV:EGU", string=True)
    read_withdrawn_unit = Cpt(EpicsSignalRO, "WVOLUME:RBV:EGU", string=True)

    infuse_rate = Cpt(EpicsSignal, "IRATE", kind="hinted")
    infuse_rate_unit = Cpt(EpicsSignal, "IRATE:EGU", string=True, kind="hinted")
    withdraw_rate = Cpt(EpicsSignal, "WRATE", kind="hinted")
    withdraw_rate_unit = Cpt(EpicsSignal, "WRATE:EGU", string=True, kind="hinted")

    read_infuse_rate = Cpt(EpicsSignalRO, "IRATE:RBV", kind="hinted")
    read_infuse_rate_unit = Cpt(
        EpicsSignalRO, "IRATE:RBV:EGU", string=True, kind="hinted"
    )
    read_withdraw_rate = Cpt(EpicsSignalRO, "WRATE:RBV", kind="hinted")
    read_withdraw_rate_unit = Cpt(
        EpicsSignalRO, "WRATE:RBV:EGU", string=True, kind="hinted"
    )

    max_infuse = Cpt(EpicsSignal, "IRATE:MAX", kind="hinted")
    max_withdraw = Cpt(EpicsSignal, "WRATE:MAX", kind="hinted")
    min_infuse = Cpt(EpicsSignal, "IRATE:MIN", kind="hinted")
    min_withdraw = Cpt(EpicsSignal, "WRATE:MIN", kind="hinted")

    set_infuse_range = Cpt(EpicsSignal, "IRATE:LIM.PROC", kind="hinted")
    set_withdraw_range = Cpt(EpicsSignal, "WRATE:LIM.PROC", kind="hinted")
    read_max_infuse = Cpt(EpicsSignalRO, "IRATE:MAX:RBV", kind="hinted")
    read_max_infuse_unit = Cpt(
        EpicsSignalRO, "IRATE:MAX:RBV:EGU", kind="hinted", string=True
    )
    read_max_withdraw = Cpt(EpicsSignalRO, "WRATE:MAX:RBV", kind="hinted")
    read_max_withdraw_unit = Cpt(
        EpicsSignalRO, "WRATE:MAX:RBV:EGU", kind="hinted", string=True
    )
    read_min_infuse = Cpt(EpicsSignalRO, "IRATE:MIN:RBV", kind="hinted")
    read_min_infuse_unit = Cpt(
        EpicsSignalRO, "IRATE:MIN:RBV:EGU", kind="hinted", string=True
    )
    read_min_withdraw = Cpt(EpicsSignalRO, "WRATE:MIN:RBV", kind="hinted")
    read_min_withdraw_unit = Cpt(
        EpicsSignalRO, "WRATE:MIN:RBV:EGU", kind="hinted", string=True
    )

    LIM = Cpt(EpicsSignal, "IRATE-LIM:ENABLE", kind="hinted")

    target_time = Cpt(EpicsSignal, "TTIME", kind="hinted")
    read_target_time = Cpt(EpicsSignalRO, "TTIME:RBV", kind="hinted")

    force = Cpt(EpicsSignal, "FORCE", kind="hinted")
    read_force = Cpt(EpicsSignalRO, "FORCE:RBV", kind="hinted")
    read_force_unit = Cpt(EpicsSignalRO, "FORCE:RBV:EGU", kind="hinted")

    # pollon = Cpt(EpicsSignal, 'POLLON', kind='config')
    # diameter = Cpt(EpicsSignalRO, 'DIRPORT:RBV', kind='config')

    def _vol_rate_table(self, syringe_material="steel"):
        min_unit = "nl/min"
        max_unit = "ml/min"

        if syringe_material == "steel":
            min_vol = {
                "2.264nl/min": 2.5,
                "8.728nl/min": 8,
                "35.21nl/min": 20,
                "78.69nl/min": 50,
            }
            _range = {
                2.5: [2.264, 2.351],
                8: [8.728, 9.064],
                20: [35.21, 36.56],
                50: [78.69, 81.72],
            }

        elif syringe_material == "glass_H1000":  # Hamilton 1000 Series
            min_vol = {
                "2.043nl/min": 1,
                "2.553nl/min": 1.25,
                "5.106nl/min": 2.5,
                "10.21nl/min": 5,
                "20.41nl/min": 10,
                "51.04nl/min": 25,
                "102.1nl/min": 50,
            }
            _range = {
                1: [2.043, 2.121],
                1.25: [2.553, 2.651],
                2.5: [5.106, 5.302],
                5: [10.21, 10.6],
                10: [20.41, 21.2],
                25: [51.04, 53],
                50: [102.1, 106],
            }

        elif syringe_material == "plastic_BD":  # Becton Dickinson
            min_vol = {
                "2.124nl/min": 1,
                "7.09nl/min": 3,
                "13.83nl/min": 5,
                "20.02nl/min": 10,
                "34.91nl/min": 20,
                "44.84nl/min": 30,
                "68.04nl/min": 50,
            }
            _range = {
                1: [2.124, 2.206],
                3: [7.09, 7.363],
                5: [13.83, 14.36],
                10: [20.02, 20.79],
                20: [34.91, 36.26],
                30: [44.84, 46.57],
                50: [68.04, 70.66],
            }

        else:
            raise ValueError(
                f"Size of {syringe_material} syringe could nor been found."
            )
        return min_vol, _range

    # def find_syringe_type(self, volume, material):
    #     if material == 'steel':
    #         #vol_dic = {'2.5': 4.851, '8': 9.525, '20': 19.13, '50': 28.6, '100':34.9}
    #         vol_min = {'2.5': 2.264, '8': 8.728, '20': 35.21, '50': 78.69}
    #         #if self.diameter.get() == vol_dic[str(volume)]:
    #         if self.read_min_infuse.get() == vol_min[str(volume)]:
    #             print(f'Selected Syringe: {volume} mL {material} syringe')
    #         else:
    #             print('Selected syringe does not fit with input. Please check.')
    #     else:
    #         print('You want to use a non-steel syringe. Please check the pump manually.')

    def reading_syringe_size(self, input_size, syringe_material):
        # dia_vol = {'4.851mm': 2.5, '9.525mm': 8, '19.13mm': 20, '28.6mm': 50, '34.9mm':100}
        self.set_infuse_range.put(1, wait=True)
        self.set_withdraw_range.put(1, wait=True)
        # min_vol= {'2.264nl/min': 2.5, '8.728nl/min': 8, '35.21nl/min': 20, '78.69nl/min': 50}
        min_vol = self._vol_rate_table(syringe_material=syringe_material)[0]
        # return dia_vol[f'{self.diameter.get()}mm']
        a = min_vol[f"{self.read_min_infuse.get()}nl/min"] == input_size
        return a, min_vol[f"{self.read_min_infuse.get()}nl/min"]

    def check_pump_condition(self, input_size, syringe_material, wait=False):
        if wait == True:
            self.set_infuse_range.put(1, wait=True)
            self.set_withdraw_range.put(1, wait=True)

        kk = self.reading_syringe_size(input_size, syringe_material)
        _, rate_range = self._vol_rate_table(syringe_material=syringe_material)
        min_rate = rate_range[input_size][0]
        max_rate = rate_range[input_size][1]
        if kk[0]:
            print("Syringe Volume: " + f"{kk[1]} mL {syringe_material} syringe")
        else:
            print(
                "(Input size doens't match the reading diameter. Use the input size.)"
            )
            print("Syringe Volume: " + f"{input_size} mL {syringe_material} syringe")
        print(f"Min rate: {min_rate} nl/min; Max rate: {max_rate} ul/min")
        print("Pump Stauts: " + f"{self.status.get()}")
        # print('Cmmunication: ' + f'{self.communication.get()} @ {self.update_pump.get()}')
        print("Cmmunication: " + f"{self.parent.communication.get()}")
        print(
            "Target Volume: "
            + f"{self.read_target_vol.get()} {self.read_target_vol_unit.get()}"
        )
        print(
            "Infuse rate: "
            + f"{self.read_infuse_rate.get()} {self.read_infuse_rate_unit.get()}"
        )
        print(
            "Withdraw rate: "
            + f"{self.read_withdraw_rate.get()} {self.read_withdraw_rate_unit.get()}"
        )
        print(
            "Infused volume: "
            + f"{self.read_infused.get()} {self.read_infused_unit.get()}"
        )
        print(
            "Withdrawn volume: "
            + f"{self.read_withdrawn.get()} {self.read_withdrawn_unit.get()}"
        )

    def initialize_pump(self, clear=True, update=".5 second"):
        self.parent.communication.put("Enabled")
        # self.parent.update_pump.put(update)

        if clear == True:
            self.clear_infused.put(1)
            self.clear_withdrawn.put(1)
        time.sleep(1)
        return self.status.get()

    def disable_pump(self, clear=True):
        if clear == True:
            self.clear_infused.put(1)
            self.clear_withdrawn.put(1)
        time.sleep(1)
        self.parent.communication.put("Disabled")
        return self.status.get()

    #     def set_pump(self, clear = False,
    #                  target_vol = 20, target_unit = 'ml',
    #                  infuse_rate = 100, infuse_unit = 'ul/min',
    #                  withdraw_rate = 100, withdraw_unit = 'ul/min'):

    #         if clear == True:
    #             self.clear_infused.put(1)
    #             self.clear_withdrawn.put(1)

    #         c = vol_unit_converter(v0=target_unit, v1='ml')
    #         if target_vol*c > self.steel_syringe_size():
    #             raise ValueError (f'Input target volume {target_vol*c} mL larger than syringe size.')
    #         self.target_vol_unit.put(target_unit, wait=True)
    #         self.target_vol.put(target_vol, wait=True)

    #         min_unit = self.show_steel_max_min_rate()[1]
    #         max_unit = self.show_steel_max_min_rate()[3]

    #         const1_max = vol_unit_converter(v0=infuse_unit[:2], v1=max_unit[:2])/t_unit_converter(t0=infuse_unit[3:], t1=max_unit[3:])
    #         const1_min = vol_unit_converter(v0=infuse_unit[:2], v1=min_unit[:2])/t_unit_converter(t0=infuse_unit[3:], t1=min_unit[3:])

    #         if infuse_rate*const1_max > self.show_steel_max_min_rate()[2]:
    #             raise ValueError(f'Input infuse rate {infuse_rate*const1_max:.3f} {max_unit} larger than allowed value.')
    #         elif infuse_rate*const1_min < self.show_steel_max_min_rate()[0]:
    #             raise ValueError(f'Input infuse rate {infuse_rate*const1_min:.3f} {min_unit} smaller than allowed value.')
    #         else:
    #             self.infuse_rate_unit.put(infuse_unit, wait=True)
    #             self.infuse_rate.put(infuse_rate, wait=True)

    #         const2_max = vol_unit_converter(v0=withdraw_unit[:2], v1=max_unit[:2])/t_unit_converter(t0=withdraw_unit[3:], t1=max_unit[3:])
    #         const2_min = vol_unit_converter(v0=withdraw_unit[:2], v1=min_unit[:2])/t_unit_converter(t0=withdraw_unit[3:], t1=min_unit[3:])
    #         if withdraw_rate*const2_max > self.show_steel_max_min_rate()[2]:
    #             raise ValueError(f'Input withdraw rate {withdraw_rate*const2_max:.3f} {max_unit} larger than allowed value.')
    #         elif withdraw_rate*const2_min < self.show_steel_max_min_rate()[0]:
    #             raise ValueError(f'Input withdraw rate {withdraw_rate*const2_min:.3f} {min_unit} smaller than allowed value.')
    #         else:
    #             self.withdraw_rate_unit.put(withdraw_unit, wait=True)
    #             self.withdraw_rate.put(withdraw_rate, wait=True)

    # def set_infuse(self, input_size, syringe_material='steel', clear = False,
    #                target_vol = 20, target_unit = 'ml',
    #                infuse_rate = 100, infuse_unit = 'ul/min'):
    #     if clear == True:
    #         yield from bps.mv(self.clear_infused, 1, self.clear_withdrawn, 1)
    #         # yield from bps.abs_set(self.clear_infused, 1, wait=True)
    #         # yield from bps.abs_set(self.clear_withdrawn, 1, wait=True)

    #     yield from bps.mv(self.set_infuse_range, 1, self.set_withdraw_range, 1)

    #     def _syringe_size(input_size, syringe_material):
    #         #yield from bps.mv(self.set_infuse_range, 1, self.set_withdraw_range, 1)
    #         min_vol = self._vol_rate_table(syringe_material=syringe_material)[0]
    #         a = (min_vol[f'{self.read_min_infuse.get()}nl/min'] == input_size)
    #         return a, min_vol[f'{self.read_min_infuse.get()}nl/min']

    #     if _syringe_size(input_size, syringe_material)[0]:
    #         size = _syringe_size(input_size, syringe_material)[1]
    #     else:
    #         #size = input_size
    #         raise ValueError('Input size could not been found. Check selected syringe type & size.')

    #     c = vol_unit_converter(v0=target_unit, v1='ml')

    #     min_unit = 'nl/min'
    #     max_unit = 'ml/min'
    #     min_in_theory = self._vol_rate_table(syringe_material=syringe_material)[1][size][0]
    #     max_in_theory = self._vol_rate_table(syringe_material=syringe_material)[1][size][1]

    #     const1_max = vol_unit_converter(v0=infuse_unit[:2], v1=max_unit[:2])/t_unit_converter(t0=infuse_unit[3:], t1=max_unit[3:])
    #     const1_min = vol_unit_converter(v0=infuse_unit[:2], v1=min_unit[:2])/t_unit_converter(t0=infuse_unit[3:], t1=min_unit[3:])

    #     if target_vol*c > size:
    #         raise ValueError (f'Input target volume {target_vol*c} mL larger than syringe size.')
    #     elif infuse_rate*const1_max > max_in_theory:
    #         raise ValueError(f'Input infuse rate {infuse_rate*const1_max:.3f} {max_unit} larger than allowed value.')
    #     elif infuse_rate*const1_min < min_in_theory:
    #         raise ValueError(f'Input infuse rate {infuse_rate*const1_min:.3f} {min_unit} smaller than allowed value.')
    #     else:
    #         yield from bps.mv(self.target_vol_unit, target_unit,
    #                           self.target_vol, target_vol,
    #                           self.infuse_rate_unit, infuse_unit,
    #                           self.infuse_rate, infuse_rate)
    #         # yield from bps.abs_set(self.infuse_rate_unit, infuse_unit, wait=True)
    #         # yield from bps.abs_set(self.infuse_rate, infuse_rate, wait=True)

    def set_infuse2(
        self,
        input_size,
        syringe_material="steel",
        clear=False,
        set_target=True,
        target_vol=20,
        target_unit="ml",
        infuse_rate=100,
        infuse_unit="ul/min",
    ):
        if clear == True:
            yield from bps.mv(self.clear_infused, 1, self.clear_withdrawn, 1)
            # yield from bps.abs_set(self.clear_infused, 1, wait=True)
            # yield from bps.abs_set(self.clear_withdrawn, 1, wait=True)

        yield from bps.mv(self.set_infuse_range, 1, self.set_withdraw_range, 1)

        def _syringe_size(input_size, syringe_material):
            # yield from bps.mv(self.set_infuse_range, 1, self.set_withdraw_range, 1)
            min_vol = self._vol_rate_table(syringe_material=syringe_material)[0]
            a = min_vol[f"{self.read_min_infuse.get()}nl/min"] == input_size
            return a, min_vol[f"{self.read_min_infuse.get()}nl/min"]

        if _syringe_size(input_size, syringe_material)[0]:
            size = _syringe_size(input_size, syringe_material)[1]
        else:
            # size = input_size
            raise ValueError(
                "Input size could not been found. Check selected syringe type & size."
            )

        if set_target:
            c = vol_unit_converter(v0=target_unit, v1="ml")
            if target_vol * c > size:
                raise ValueError(
                    f"Input target volume {target_vol * c} mL larger than syringe size."
                )
            yield from bps.abs_set(self.target_vol_unit, target_unit, wait=True)
            yield from bps.abs_set(self.target_vol, target_vol, wait=True)

        min_unit = "nl/min"
        max_unit = "ml/min"
        min_in_device = self._vol_rate_table(syringe_material=syringe_material)[1][
            size
        ][0]
        max_in_device = self._vol_rate_table(syringe_material=syringe_material)[1][
            size
        ][1]

        # const1_max = vol_unit_converter(v0=infuse_unit[:2], v1=max_unit[:2])/t_unit_converter(t0=infuse_unit[3:], t1=max_unit[3:])
        # const1_min = vol_unit_converter(v0=infuse_unit[:2], v1=min_unit[:2])/t_unit_converter(t0=infuse_unit[3:], t1=min_unit[3:])
        const1_max = rate_unit_converter(r0=infuse_unit, r1=max_unit)
        const1_min = rate_unit_converter(r0=infuse_unit, r1=min_unit)

        if infuse_rate * const1_max > max_in_device:
            raise ValueError(
                f"Input infuse rate {infuse_rate * const1_max:.3f} {max_unit} larger than allowed value."
            )
        elif infuse_rate * const1_min < min_in_device:
            raise ValueError(
                f"Input infuse rate {infuse_rate * const1_min:.3f} {min_unit} smaller than allowed value."
            )

        yield from bps.abs_set(self.infuse_rate_unit, infuse_unit, wait=True)
        yield from bps.abs_set(self.infuse_rate, infuse_rate, wait=True)
        yield from bps.sleep(1)

    def set_withdraw(
        self,
        input_size,
        syringe_material="steel",
        clear=False,
        target_vol=20,
        target_unit="ml",
        withdraw_rate=100,
        withdraw_unit="ul/min",
    ):
        if clear == True:
            yield from bps.mv(self.clear_infused, 1, self.clear_withdrawn, 1)
            # yield from bps.abs_set(self.clear_infused, 1, wait=True)
            # yield from bps.abs_set(self.clear_withdrawn, 1, wait=True)

        yield from bps.mv(self.set_infuse_range, 1, self.set_withdraw_range, 1)

        def _syringe_size(input_size, syringe_material):
            # yield from bps.mv(self.set_infuse_range, 1, self.set_withdraw_range, 1)
            min_vol = set._vol_rate_table(syringe_material=syringe_material)[0]
            a = min_vol[f"{self.read_min_infuse.get()}nl/min"] == input_size
            return a, min_vol[f"{self.read_min_infuse.get()}nl/min"]

        if _syringe_size(input_size, syringe_material)[0]:
            size = _syringe_size(input_size, syringe_material)[1]
        else:
            # size = input_size
            raise ValueError(
                "Input size could not been found. Check selected syringe type & size."
            )

        c = vol_unit_converter(v0=target_unit, v1="ml")

        min_unit = "nl/min"
        max_unit = "ml/min"
        min_in_theory = set._vol_rate_table(syringe_material=syringe_material)[1][size][
            0
        ]
        max_in_theory = set._vol_rate_table(syringe_material=syringe_material)[1][size][
            1
        ]

        const1_max = vol_unit_converter(
            v0=withdraw_unit[:2], v1=max_unit[:2]
        ) / t_unit_converter(t0=withdraw_unit[3:], t1=max_unit[3:])
        const1_min = vol_unit_converter(
            v0=withdraw_unit[:2], v1=min_unit[:2]
        ) / t_unit_converter(t0=withdraw_unit[3:], t1=min_unit[3:])

        if target_vol * c > size:
            raise ValueError(
                f"Input target volume {target_vol * c} mL larger than syringe size."
            )
        elif withdraw_rate * const1_max > max_in_theory:
            raise ValueError(
                f"Input withdraw rate {withdraw_rate * const1_max:.3f} {max_unit} larger than allowed value."
            )
        elif withdraw_rate * const1_min < min_in_theory:
            raise ValueError(
                f"Input withdraw rate {withdraw_rate * const1_min:.3f} {min_unit} smaller than allowed value."
            )
        else:
            yield from bps.mv(
                self.target_vol_unit,
                target_unit,
                self.target_vol,
                target_vol,
                self.withdraw_rate_unit,
                withdraw_unit,
                self.withdraw_rate,
                withdraw_rate,
            )
            # yield from bps.abs_set(self.infuse_rate_unit, infuse_unit, wait=True)
            # yield from bps.abs_set(self.infuse_rate, infuse_rate, wait=True)

    def infuse_pump(self, clear=False):
        if clear == True:
            self.clear_infused.put(1)
            self.clear_withdrawn.put(1)
        self.pump_infuse.put(1)
        time.sleep(1)
        return self.status.get()

    def infuse_pump2(self, clear=False):
        if clear == True:
            yield from bps.mv(self.clear_infused, 1, self.clear_withdrawn, 1)
            # yield from bps.abs_set(self.clear_infused, 1, wait=True)
            # yield from bps.abs_set(self.clear_withdrawn, 1, wait=True)
        yield from bps.abs_set(self.pump_infuse, 1, wait=True)
        yield from bps.sleep(1)

        # time.sleep(1)
        ## There is a specific way in RE{}; time.sleep is dangerous for RE{} --> use bps.sleep()
        # return self.status.get()

    def withdraw_pump(self, clear=False):
        if clear == True:
            self.clear_infused.put(1)
            self.clear_withdrawn.put(1)
        self.pump_withdraw.put(1)
        time.sleep(1)
        return self.status.get()

    def withdraw_pump2(self, clear=False):
        if clear == True:
            yield from bps.mv(self.clear_infused, 1, self.clear_withdrawn, 1)
            # yield from bps.abs_set(self.clear_infused, 1, wait=True)
            # yield from bps.abs_set(self.clear_withdrawn, 1, wait=True)
        yield from bps.abs_set(self.pump_withdraw, 1, wait=True)
        yield from bps.sleep(1)

    def stop_pump(self, clear=False):
        self.pump_stop.put(1)
        if clear == True:
            self.clear_infused.put(1)
            self.clear_withdrawn.put(1)
        time.sleep(1)
        return self.status.get()

    def stop_pump2(self, clear=False):
        yield from bps.abs_set(self.pump_stop, 1, wait=True)
        if clear == True:
            yield from bps.mv(self.clear_infused, 1, self.clear_withdrawn, 1)
            # yield from bps.abs_set(self.clear_infused, 1, wait=True)
            # yield from bps.abs_set(self.clear_withdrawn, 1, wait=True)
        yield from bps.sleep(1)

    def show_steel_max_min_rate(self, input_size):
        min_unit = "nl/min"
        max_unit = "ml/min"

        if self.reading_syringe_size(input_size)[0]:
            size = self.reading_syringe_size(input_size)[1]
        else:
            size = input_size

        if size == 2.5:
            min_rate = 2.264
            max_rate = 2.351
        elif size == 8:
            min_rate = 8.728
            max_rate = 9.064
        elif size == 20:
            min_rate = 35.21
            max_rate = 36.56
        elif size == 50:
            min_rate = 78.69
            max_rate = 81.72
        # elif self.steel_syringe_size() == 100:
        #    min_rate = 175.769
        #    max_rate = 182.529
        else:
            min_rate = "Unkonwn"
            ax_rate = "Unknown"
        return min_rate, min_unit, max_rate, max_unit


def _dds_vol_rate_table(syringe_material="steel"):
    min_unit = "nl/min"
    max_unit = "ml/min"

    if syringe_material == "steel":
        min_vol = {
            "2.264nl/min": 2.5,
            "8.728nl/min": 8,
            "35.21nl/min": 20,
            "78.69nl/min": 50,
        }
        _range = {
            2.5: [2.264, 2.351],
            8: [8.728, 9.064],
            20: [35.21, 36.56],
            50: [78.69, 81.72],
        }

    elif syringe_material == "glass_H1000":  # Hamilton 1000 Series
        min_vol = {
            "2.043nl/min": 1,
            "2.553nl/min": 1.25,
            "5.106nl/min": 2.5,
            "10.21nl/min": 5,
            "20.41nl/min": 10,
            "51.04nl/min": 25,
            "102.1nl/min": 50,
        }
        _range = {
            1: [2.043, 2.121],
            1.25: [2.553, 2.651],
            2.5: [5.106, 5.302],
            5: [10.21, 10.6],
            10: [20.41, 21.2],
            25: [51.04, 53],
            50: [102.1, 106],
        }

    elif syringe_material == "plastic_BD":  # Becton Dickinson
        min_vol = {
            "2.124nl/min": 1,
            "7.09nl/min": 3,
            "13.83nl/min": 5,
            "20.02nl/min": 10,
            "34.91nl/min": 20,
            "44.84nl/min": 30,
            "68.04nl/min": 50,
        }
        _range = {
            1: [2.124, 2.206],
            3: [7.09, 7.363],
            5: [13.83, 14.36],
            10: [20.02, 20.79],
            20: [34.91, 36.26],
            30: [44.84, 46.57],
            50: [68.04, 70.66],
        }

    else:
        raise ValueError(f"Size of {syringe_material} syringe could nor been found.")
    return min_vol, _range


def vol_unit_converter(v0="ul", v1="ml"):
    vol_unit = ["pl", "nl", "ul", "ml"]
    vol_frame = pd.DataFrame(
        data={
            "pl": np.geomspace(1, 1e9, num=4),
            "nl": np.geomspace(1e-3, 1e6, num=4),
            "ul": np.geomspace(1e-6, 1e3, num=4),
            "ml": np.geomspace(1e-9, 1, num=4),
        },
        index=vol_unit,
    )
    return vol_frame.loc[v0, v1]


def t_unit_converter(t0="min", t1="min"):
    t_unit = ["sec", "min", "hr"]
    t_frame = pd.DataFrame(
        data={
            "sec": np.geomspace(1, 3600, num=3),
            "min": np.geomspace(1 / 60, 60, num=3),
            "hr": np.geomspace(1 / 3600, 1, num=3),
        },
        index=t_unit,
    )
    return t_frame.loc[t0, t1]


def rate_unit_converter(r0="ul/min", r1="ul/min"):

    v0 = r0.split("/")[0]
    t0 = r0.split("/")[1]
    v1 = r1.split("/")[0]
    t1 = r1.split("/")[1]

    ## ruc = rate_unit_converter
    ruc = vol_unit_converter(v0=v0, v1=v1) / t_unit_converter(t0=t0, t1=t1)
    return ruc


def syringe_diameter(volume, material="steel"):
    if material == "steel":
        vol_dic = {"2.5": 4.851, "8": 9.525, "20": 19.13, "50": 28.6, "100": 34.9}
        if str(volume) in vol_dic:
            print(
                f"{volume} mL {material} Syringe Diameter is {vol_dic[str(volume)]} mm."
            )
        else:
            raise ValueError(
                "Input volume is not supported by Harvard stainless steel syringe."
            )
    else:
        print("Please check the diameter of non-steel syringe manually.")


# dds1 = syrng_DDS('XF:28IDC-ES:1{Pump:Syrng-DDS:1', name='DDS1')

# dds1_p1 = syrng_DDS_ax('XF:28IDC-ES:1{Pump:Syrng-DDS:1-Ax:A}', name='DDS1_p1', parent=dds1,
#                         read_attrs=['status',
#                                     'read_target_vol', 'read_target_vol_unit',
#                                     'read_infused', 'read_infused_unit',
#                                     'read_withdrawn', 'read_withdrawn_unit',
#                                     'read_infuse_rate', 'read_infuse_rate_unit',
#                                     'read_withdraw_rate', 'read_withdraw_rate_unit'])

# dds1_p2 = syrng_DDS_ax('XF:28IDC-ES:1{Pump:Syrng-DDS:1-Ax:B}', name='DDS1_p2', parent=dds1,
#                         read_attrs=['status',
#                                     'read_target_vol', 'read_target_vol_unit',
#                                     'read_infused', 'read_infused_unit',
#                                     'read_withdrawn', 'read_withdrawn_unit',
#                                     'read_infuse_rate', 'read_infuse_rate_unit',
#                                     'read_withdraw_rate', 'read_withdraw_rate_unit'])


# dds2 = syrng_DDS('XF:28IDC-ES:1{Pump:Syrng-DDS:2', name='DDS2')

# dds2_p1 = syrng_DDS_ax('XF:28IDC-ES:1{Pump:Syrng-DDS:2-Ax:A}', name='DDS2_p1', parent=dds2,
#                      read_attrs=['status',
#                                  'read_target_vol', 'read_target_vol_unit',
#                                  'read_infused', 'read_infused_unit',
#                                  'read_withdrawn', 'read_withdrawn_unit',
#                                  'read_infuse_rate', 'read_infuse_rate_unit',
#                                  'read_withdraw_rate', 'read_withdraw_rate_unit'])

# dds2_p2 = syrng_DDS_ax('XF:28IDC-ES:1{Pump:Syrng-DDS:2-Ax:B}', name='DDS2_p2', parent=dds2,
#                      read_attrs=['status',
#                                  'read_target_vol', 'read_target_vol_unit',
#                                  'read_infused', 'read_infused_unit',
#                                  'read_withdrawn', 'read_withdrawn_unit',
#                                  'read_infuse_rate', 'read_infuse_rate_unit',
#                                  'read_withdraw_rate', 'read_withdraw_rate_unit'])

# XF:28IDC-ES:1{Pump:Syrng-DDS:3-Ax:A}STATUS

# dds3 = syrng_DDS('XF:28IDC-ES:1{Pump:Syrng-DDS:3', name='DDS3')

# dds3_p1 = syrng_DDS_ax('XF:28IDC-ES:1{Pump:Syrng-DDS:3-Ax:A}', name='DDS3_p1', parent=dds3,
#                      read_attrs=['status',
#                                  'read_target_vol', 'read_target_vol_unit',
#                                  'read_infused', 'read_infused_unit',
#                                  'read_withdrawn', 'read_withdrawn_unit',
#                                  'read_infuse_rate', 'read_infuse_rate_unit',
#                                  'read_withdraw_rate', 'read_withdraw_rate_unit'])

# dds3_p2 = syrng_DDS_ax('XF:28IDC-ES:1{Pump:Syrng-DDS:3-Ax:B}', name='DDS3_p2', parent=dds3,
#                      read_attrs=['status',
#                                  'read_target_vol', 'read_target_vol_unit',
#                                  'read_infused', 'read_infused_unit',
#                                  'read_withdrawn', 'read_withdrawn_unit',
#                                  'read_infuse_rate', 'read_infuse_rate_unit',
#                                  'read_withdraw_rate', 'read_withdraw_rate_unit'])

# dds1_p2 = syrng_DDS_ax('XF:28IDC-ES:1{Pump:Syrng-DDS:1-Ax:B}', name='DDS1_p2', parent=dds1,
#                         read_attrs=['status',
#                                     'read_target_vol', 'read_target_vol_unit',
#                                     'read_infused', 'read_infused_unit',
#                                     'read_withdrawn', 'read_withdrawn_unit',
#                                     'read_infuse_rate', 'read_infuse_rate_unit',
#                                     'read_withdraw_rate', 'read_withdraw_rate_unit'])
#
#
#
#
# dds2 = syrng_DDS('XF:28IDC-ES:1{Pump:Syrng-DDS:2', name='DDS2')
#
# dds2_p1 = syrng_DDS_ax('XF:28IDC-ES:1{Pump:Syrng-DDS:2-Ax:A}', name='DDS2_p1', parent=dds2,
#                      read_attrs=['status',
#                                  'read_target_vol', 'read_target_vol_unit',
#                                  'read_infused', 'read_infused_unit',
#                                  'read_withdrawn', 'read_withdrawn_unit',
#                                  'read_infuse_rate', 'read_infuse_rate_unit',
#                                  'read_withdraw_rate', 'read_withdraw_rate_unit'])
#
# dds2_p2 = syrng_DDS_ax('XF:28IDC-ES:1{Pump:Syrng-DDS:2-Ax:B}', name='DDS2_p2', parent=dds2,
#                      read_attrs=['status',
#                                  'read_target_vol', 'read_target_vol_unit',
#                                  'read_infused', 'read_infused_unit',
#                                  'read_withdrawn', 'read_withdrawn_unit',
#                                  'read_infuse_rate', 'read_infuse_rate_unit',
#                                  'read_withdraw_rate', 'read_withdraw_rate_unit'])
#
# # XF:28IDC-ES:1{Pump:Syrng-DDS:3-Ax:A}STATUS
#
# dds3 = syrng_DDS('XF:28IDC-ES:1{Pump:Syrng-DDS:3', name='DDS3')
#
# dds3_p1 = syrng_DDS_ax('XF:28IDC-ES:1{Pump:Syrng-DDS:3-Ax:A}', name='DDS3_p1', parent=dds3,
#                      read_attrs=['status',
#                                  'read_target_vol', 'read_target_vol_unit',
#                                  'read_infused', 'read_infused_unit',
#                                  'read_withdrawn', 'read_withdrawn_unit',
#                                  'read_infuse_rate', 'read_infuse_rate_unit',
#                                  'read_withdraw_rate', 'read_withdraw_rate_unit'])
#
# dds3_p2 = syrng_DDS_ax('XF:28IDC-ES:1{Pump:Syrng-DDS:3-Ax:B}', name='DDS3_p2', parent=dds3,
#                      read_attrs=['status',
#                                  'read_target_vol', 'read_target_vol_unit',
#                                  'read_infused', 'read_infused_unit',
#                                  'read_withdrawn', 'read_withdrawn_unit',
#                                  'read_infuse_rate', 'read_infuse_rate_unit',
#                                  'read_withdraw_rate', 'read_withdraw_rate_unit'])
