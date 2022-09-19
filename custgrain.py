from __future__ import annotations
# from enum import IntEnum
from naunet.network import define_reaction, define_grain
from naunet.reactions import Reaction, UCLCHEMReaction
from naunet.component import VariableType as vt
from naunet.grains.rr07grain import RR07XGrain
from naunet.species import Species


@define_reaction("uclcustom")
class CUSTOMReaction(UCLCHEMReaction):

    def __init__(self, react_string: str) -> None:
        super().__init__(react_string=react_string)

    
    def _parse_string(self, react_string) -> None:

        super()._parse_string(react_string)
        if self.reaction_type != self.ReactionType.UCLCHEM_FR:
            self.temp_min = -1.0
            self.temp_max = -1.0

@define_grain("rr07custom")
class CUSTOMGrain(RR07XGrain):

    def __init__(self, species: list[Species] = None, group: int = 0) -> None:
        super().__init__(species, group)
        group = group or ""
        self.register("sputtering_rate", (f"ksp{group}", 0.0, vt.param))
        self.register("habing_field_photon_number", ("habing", 1e8, vt.constant))
        self.register("cosmic_ray_induced_photon_number", ("crphot", 1e4, vt.constant))
        self.register("active_monolayer_number", (f"nmono{group}", 2.0, vt.constant))

    def rate_thermal_desorption(self, reac: Reaction) -> str:

        super().rate_thermal_desorption(reac)

        tdust = reac.symbols.dust_temperature.symbol

        mant = self.symbols.mantle_number_density.symbol
        mantabund = self.symbols.mantle_number_density_per_H.symbol
        sites = self.symbols.grain_surface_sites_density.symbol
        densites = self.symbols.surface_sites_density.symbol
        opt_thd = self.symbols.thermal_desorption_option.symbol
        gxsec = self.symbols.grain_cross_section.symbol
        ksp = self.symbols.sputtering_rate.symbol

        spec = reac.reactants[0]
        rate = " * ".join(
            [
                f"{opt_thd}",
                f"sqrt(2.0*{sites}*kerg*eb_{spec.alias}/(pi*pi*amu*{spec.massnumber}))",
                f"2.0 * {densites}",
                f"exp(-eb_{spec.alias}/{tdust})",
            ],
        )

        rate = f"({rate} + {ksp} * {gxsec} / {mant})"
        rate = f"{mantabund} > 1e-30 ? ({rate}) : 0.0"
        return rate

    def rate_photon_desorption(self, reac: Reaction) -> str:

        super().rate_photon_desorption(reac)

        crrate = reac.symbols.cosmic_ray_ionization_rate.symbol
        zism = reac.symbols.ism_cosmic_ray_ionization_rate.symbol
        radfield = reac.symbols.radiation_field.symbol
        av = reac.symbols.visual_extinction.symbol

        habing = self.symbols.habing_field_photon_number.symbol
        crphot = self.symbols.cosmic_ray_induced_photon_number.symbol
        opt_uvd = self.symbols.photon_desorption_option.symbol
        nmono = self.symbols.active_monolayer_number.symbol
        gxsec = self.symbols.grain_cross_section.symbol

        sym_phot = f"{radfield}*{habing}*exp(-{av}*3.02) + {crphot} * ({crrate}/{zism})"

        spec = reac.reactants[0]
        rate = f"{opt_uvd} * ({sym_phot}) * {spec.photon_yield()} * {nmono} * 4.0 * {gxsec}"
        rate = f"mantabund > 1e-30 ? ({rate}) : 0.0"

        return rate
