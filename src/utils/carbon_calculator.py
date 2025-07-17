"""
Carbon footprint calculation utilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class CarbonFootprintCalculator:
    """Calculate carbon footprint for products and processes"""
    
    def __init__(self):
        self.emission_factors = self._load_emission_factors()
        self.material_factors = self._load_material_factors()
        self.transportation_factors = self._load_transportation_factors()
        self.energy_factors = self._load_energy_factors()
    
    def _load_emission_factors(self) -> Dict[str, float]:
        """Load emission factors for different categories"""
        return {
            'manufacturing': 2.5,  # kg CO2e per kg product
            'packaging': 1.2,      # kg CO2e per kg packaging
            'disposal': 0.8,       # kg CO2e per kg waste
            'recycling': -0.5,     # kg CO2e saved per kg recycled
            'electricity': 0.45,   # kg CO2e per kWh
            'natural_gas': 0.2,    # kg CO2e per kWh
            'water': 0.0003       # kg CO2e per liter
        }
    
    def _load_material_factors(self) -> Dict[str, float]:
        """Load carbon intensity factors for different materials"""
        return {
            'steel': 2.9,         # kg CO2e per kg
            'aluminum': 11.5,     # kg CO2e per kg
            'plastic': 3.4,       # kg CO2e per kg
            'glass': 0.85,        # kg CO2e per kg
            'paper': 1.1,         # kg CO2e per kg
            'wood': 0.72,         # kg CO2e per kg
            'cotton': 8.0,        # kg CO2e per kg
            'polyester': 9.5,     # kg CO2e per kg
            'concrete': 0.93,     # kg CO2e per kg
            'cardboard': 0.9      # kg CO2e per kg
        }
    
    def _load_transportation_factors(self) -> Dict[str, float]:
        """Load transportation emission factors"""
        return {
            'truck': 0.16,        # kg CO2e per km per kg
            'ship': 0.014,        # kg CO2e per km per kg
            'plane': 0.5,         # kg CO2e per km per kg
            'train': 0.03,        # kg CO2e per km per kg
            'local_delivery': 0.2  # kg CO2e per km per kg
        }
    
    def _load_energy_factors(self) -> Dict[str, float]:
        """Load energy source emission factors"""
        return {
            'coal': 0.82,         # kg CO2e per kWh
            'natural_gas': 0.49,  # kg CO2e per kWh
            'oil': 0.78,          # kg CO2e per kWh
            'nuclear': 0.012,     # kg CO2e per kWh
            'hydro': 0.024,       # kg CO2e per kWh
            'wind': 0.011,        # kg CO2e per kWh
            'solar': 0.041,       # kg CO2e per kWh
            'biomass': 0.23,      # kg CO2e per kWh
            'geothermal': 0.038   # kg CO2e per kWh
        }
    
    def calculate_product_footprint(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate carbon footprint for a single product
        
        Args:
            product_data: Product information dictionary
            
        Returns:
            Carbon footprint calculation results
        """
        footprint_components = {
            'materials': 0.0,
            'manufacturing': 0.0,
            'packaging': 0.0,
            'transportation': 0.0,
            'energy': 0.0,
            'end_of_life': 0.0
        }
        
        # Material footprint
        materials = product_data.get('materials', {})
        for material, weight in materials.items():
            if material in self.material_factors:
                footprint_components['materials'] += weight * self.material_factors[material]
        
        # Manufacturing footprint
        manufacturing_energy = product_data.get('manufacturing_energy', 0)  # kWh
        energy_mix = product_data.get('energy_mix', {'electricity': 1.0})
        
        for energy_type, proportion in energy_mix.items():
            if energy_type in self.energy_factors:
                footprint_components['manufacturing'] += (
                    manufacturing_energy * proportion * self.energy_factors[energy_type]
                )
        
        # Packaging footprint
        packaging_materials = product_data.get('packaging_materials', {})
        for material, weight in packaging_materials.items():
            if material in self.material_factors:
                footprint_components['packaging'] += weight * self.material_factors[material]
        
        # Transportation footprint
        transportation = product_data.get('transportation', [])
        for transport in transportation:
            distance = transport.get('distance', 0)  # km
            weight = transport.get('weight', product_data.get('weight', 1))  # kg
            mode = transport.get('mode', 'truck')
            
            if mode in self.transportation_factors:
                footprint_components['transportation'] += (
                    distance * weight * self.transportation_factors[mode]
                )
        
        # Energy consumption during use
        use_energy = product_data.get('use_energy_kwh', 0)
        use_duration = product_data.get('use_duration_years', 1)
        
        if use_energy > 0:
            annual_energy = use_energy * 365  # Assuming daily usage
            total_use_energy = annual_energy * use_duration
            
            user_energy_mix = product_data.get('user_energy_mix', {'electricity': 1.0})
            for energy_type, proportion in user_energy_mix.items():
                if energy_type in self.energy_factors:
                    footprint_components['energy'] += (
                        total_use_energy * proportion * self.energy_factors[energy_type]
                    )
        
        # End-of-life footprint
        weight = product_data.get('weight', 1)
        recycling_rate = product_data.get('recycling_rate', 0)  # 0-1
        
        # Disposal emissions
        footprint_components['end_of_life'] += weight * (1 - recycling_rate) * self.emission_factors['disposal']
        
        # Recycling credits
        footprint_components['end_of_life'] += weight * recycling_rate * self.emission_factors['recycling']
        
        # Calculate total footprint
        total_footprint = sum(footprint_components.values())
        
        return {
            'total_footprint_kg_co2e': total_footprint,
            'components': footprint_components,
            'per_unit_footprint': total_footprint / product_data.get('quantity', 1),
            'breakdown_percentage': {
                component: (value / total_footprint * 100) if total_footprint > 0 else 0
                for component, value in footprint_components.items()
            },
            'calculation_date': datetime.now().isoformat()
        }
    
    def calculate_basket_footprint(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate carbon footprint for a basket of products
        
        Args:
            products: List of product dictionaries
            
        Returns:
            Basket carbon footprint results
        """
        total_footprint = 0.0
        total_components = {
            'materials': 0.0,
            'manufacturing': 0.0,
            'packaging': 0.0,
            'transportation': 0.0,
            'energy': 0.0,
            'end_of_life': 0.0
        }
        
        product_footprints = []
        
        for product in products:
            quantity = product.get('quantity', 1)
            product_result = self.calculate_product_footprint(product)
            
            # Scale by quantity
            product_footprint = product_result['total_footprint_kg_co2e'] * quantity
            total_footprint += product_footprint
            
            # Add to total components
            for component, value in product_result['components'].items():
                total_components[component] += value * quantity
            
            product_footprints.append({
                'product_id': product.get('id', 'unknown'),
                'product_name': product.get('name', 'Unknown Product'),
                'quantity': quantity,
                'footprint_kg_co2e': product_footprint,
                'components': {k: v * quantity for k, v in product_result['components'].items()}
            })
        
        # Generate comparisons
        comparisons = self._generate_comparisons(total_footprint)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(product_footprints, total_components)
        
        return {
            'total_footprint_kg_co2e': total_footprint,
            'components': total_components,
            'product_footprints': product_footprints,
            'comparisons': comparisons,
            'recommendations': recommendations,
            'calculation_date': datetime.now().isoformat()
        }
    
    def _generate_comparisons(self, footprint_kg_co2e: float) -> Dict[str, Any]:
        """Generate relatable comparisons for carbon footprint"""
        return {
            'car_miles_equivalent': footprint_kg_co2e * 2.5,  # miles
            'ac_usage_hours': footprint_kg_co2e * 0.5,       # hours
            'trees_to_offset': footprint_kg_co2e * 0.04,     # trees
            'phone_charges': footprint_kg_co2e * 121,        # charges
            'light_bulb_hours': footprint_kg_co2e * 14,      # hours
            'shower_minutes': footprint_kg_co2e * 2.3        # minutes
        }
    
    def _generate_recommendations(self, product_footprints: List[Dict[str, Any]], 
                                total_components: Dict[str, float]) -> List[str]:
        """Generate recommendations to reduce carbon footprint"""
        recommendations = []
        
        # Find highest impact component
        max_component = max(total_components.items(), key=lambda x: x[1])
        
        if max_component[0] == 'materials':
            recommendations.append("Consider products with lower-impact materials")
        elif max_component[0] == 'transportation':
            recommendations.append("Choose locally sourced products to reduce transportation emissions")
        elif max_component[0] == 'energy':
            recommendations.append("Look for energy-efficient alternatives")
        elif max_component[0] == 'manufacturing':
            recommendations.append("Select products from manufacturers using renewable energy")
        
        # Find highest impact products
        if product_footprints:
            highest_impact = max(product_footprints, key=lambda x: x['footprint_kg_co2e'])
            recommendations.append(f"Consider alternatives to {highest_impact['product_name']} (highest carbon impact)")
        
        # General recommendations
        recommendations.extend([
            "Offset remaining emissions through verified carbon credits",
            "Choose products with recycled content",
            "Consider product longevity and repairability",
            "Look for products with minimal packaging"
        ])
        
        return recommendations
    
    def calculate_supply_chain_footprint(self, supply_chain_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate carbon footprint for entire supply chain
        
        Args:
            supply_chain_data: Supply chain information
            
        Returns:
            Supply chain carbon footprint results
        """
        stages = supply_chain_data.get('stages', [])
        total_footprint = 0.0
        stage_footprints = []
        
        for stage in stages:
            stage_footprint = 0.0
            
            # Raw material extraction
            if stage.get('type') == 'raw_material':
                materials = stage.get('materials', {})
                for material, quantity in materials.items():
                    if material in self.material_factors:
                        stage_footprint += quantity * self.material_factors[material]
            
            # Manufacturing
            elif stage.get('type') == 'manufacturing':
                energy_consumption = stage.get('energy_consumption', 0)
                energy_mix = stage.get('energy_mix', {'electricity': 1.0})
                
                for energy_type, proportion in energy_mix.items():
                    if energy_type in self.energy_factors:
                        stage_footprint += energy_consumption * proportion * self.energy_factors[energy_type]
            
            # Transportation
            elif stage.get('type') == 'transportation':
                distance = stage.get('distance', 0)
                weight = stage.get('weight', 1)
                mode = stage.get('mode', 'truck')
                
                if mode in self.transportation_factors:
                    stage_footprint += distance * weight * self.transportation_factors[mode]
            
            total_footprint += stage_footprint
            stage_footprints.append({
                'stage_name': stage.get('name', 'Unknown Stage'),
                'stage_type': stage.get('type', 'unknown'),
                'footprint_kg_co2e': stage_footprint,
                'percentage': 0  # Will be calculated after total is known
            })
        
        # Calculate percentages
        for stage in stage_footprints:
            stage['percentage'] = (stage['footprint_kg_co2e'] / total_footprint * 100) if total_footprint > 0 else 0
        
        return {
            'total_supply_chain_footprint_kg_co2e': total_footprint,
            'stages': stage_footprints,
            'hotspots': self._identify_hotspots(stage_footprints),
            'calculation_date': datetime.now().isoformat()
        }
    
    def _identify_hotspots(self, stage_footprints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify carbon hotspots in supply chain"""
        # Sort by footprint
        sorted_stages = sorted(stage_footprints, key=lambda x: x['footprint_kg_co2e'], reverse=True)
        
        # Identify stages contributing more than 20% of total emissions
        hotspots = [stage for stage in sorted_stages if stage['percentage'] > 20]
        
        return hotspots
    
    def calculate_carbon_intensity(self, production_data: Dict[str, Any]) -> float:
        """
        Calculate carbon intensity (kg CO2e per unit output)
        
        Args:
            production_data: Production information
            
        Returns:
            Carbon intensity value
        """
        total_emissions = production_data.get('total_emissions_kg_co2e', 0)
        production_volume = production_data.get('production_volume', 1)
        
        return total_emissions / production_volume if production_volume > 0 else 0

class CarbonOffsetCalculator:
    """Calculate carbon offset requirements and costs"""
    
    def __init__(self):
        self.offset_prices = self._load_offset_prices()
        self.offset_projects = self._load_offset_projects()
    
    def _load_offset_prices(self) -> Dict[str, float]:
        """Load carbon offset prices by project type"""
        return {
            'forestry': 12.0,        # USD per tonne CO2e
            'renewable_energy': 8.0,  # USD per tonne CO2e
            'methane_capture': 15.0,  # USD per tonne CO2e
            'direct_air_capture': 150.0,  # USD per tonne CO2e
            'improved_cookstoves': 5.0,   # USD per tonne CO2e
            'soil_carbon': 20.0      # USD per tonne CO2e
        }
    
    def _load_offset_projects(self) -> List[Dict[str, Any]]:
        """Load available offset projects"""
        return [
            {
                'name': 'Amazon Rainforest Protection',
                'type': 'forestry',
                'location': 'Brazil',
                'certification': 'VCS',
                'price_per_tonne': 12.0,
                'co_benefits': ['biodiversity', 'community_development']
            },
            {
                'name': 'Wind Farm Development',
                'type': 'renewable_energy',
                'location': 'India',
                'certification': 'Gold Standard',
                'price_per_tonne': 8.0,
                'co_benefits': ['job_creation', 'air_quality']
            },
            {
                'name': 'Methane Capture from Landfills',
                'type': 'methane_capture',
                'location': 'USA',
                'certification': 'CAR',
                'price_per_tonne': 15.0,
                'co_benefits': ['waste_management', 'air_quality']
            }
        ]
    
    def calculate_offset_requirements(self, carbon_footprint_kg_co2e: float,
                                   offset_percentage: float = 100) -> Dict[str, Any]:
        """
        Calculate carbon offset requirements
        
        Args:
            carbon_footprint_kg_co2e: Carbon footprint in kg CO2e
            offset_percentage: Percentage of emissions to offset (0-100)
            
        Returns:
            Offset requirements and costs
        """
        # Convert to tonnes
        carbon_footprint_tonnes = carbon_footprint_kg_co2e / 1000
        
        # Calculate offset requirement
        offset_requirement = carbon_footprint_tonnes * (offset_percentage / 100)
        
        # Calculate costs for different project types
        offset_costs = {}
        for project_type, price in self.offset_prices.items():
            offset_costs[project_type] = offset_requirement * price
        
        # Find recommended projects
        recommended_projects = self._recommend_projects(offset_requirement)
        
        return {
            'carbon_footprint_tonnes_co2e': carbon_footprint_tonnes,
            'offset_requirement_tonnes': offset_requirement,
            'offset_percentage': offset_percentage,
            'estimated_costs_usd': offset_costs,
            'recommended_projects': recommended_projects,
            'calculation_date': datetime.now().isoformat()
        }
    
    def _recommend_projects(self, offset_requirement: float) -> List[Dict[str, Any]]:
        """Recommend offset projects based on requirements"""
        # Sort projects by price and co-benefits
        sorted_projects = sorted(self.offset_projects, key=lambda x: x['price_per_tonne'])
        
        recommendations = []
        for project in sorted_projects[:3]:  # Top 3 recommendations
            cost = offset_requirement * project['price_per_tonne']
            recommendations.append({
                'project': project,
                'cost_usd': cost,
                'tonnes_offset': offset_requirement
            })
        
        return recommendations

# Global calculator instances
carbon_calculator = CarbonFootprintCalculator()
offset_calculator = CarbonOffsetCalculator()
