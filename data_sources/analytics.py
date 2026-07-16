"""
analytics.py — deterministic financial analytics. No LLM involved.

1. normalize_xbrl_facts(): pulls the metrics we care about out of SEC's raw
   Company Facts XBRL payload, resolving alternate US-GAAP tags and selecting
   sensible values per reporting period.

2. calculate_ratios(): computes standard ratios from compatible reporting
   periods only. Growth is calculated only between comparable prior-year
   periods. Cross-period ratios are calculated only when the underlying
   duration metrics cover compatible periods.

Missing or non-comparable data returns None — never a guess and never zero.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Optional

from .models import FinancialMetric


logger = logging.getLogger("finbot.analytics")


# ---------------------------------------------------------------------------
# US-GAAP concept mapping
# ---------------------------------------------------------------------------

# Metric -> ordered list of US-GAAP concept names to try,
# most-preferred first.
METRIC_MAP: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "Revenues",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
    ],
    "gross_profit": [
        "GrossProfit",
    ],
    "total_assets": [
        "Assets",
    ],
    "current_assets": [
        "AssetsCurrent",
    ],
    "current_liabilities": [
        "LiabilitiesCurrent",
    ],
    "cash_and_cash_equivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "total_debt": [
        "DebtLongtermAndShorttermCombinedAmount",
        "LongTermDebtNoncurrent",
    ],
    "stockholders_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
    ],
    "capital_expenditure": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
    "diluted_eps": [
        "EarningsPerShareDiluted",
    ],
    "shares_outstanding": [
        "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding",
    ],
}


# Metrics that are point-in-time balances rather than duration facts.
INSTANT_METRICS = {
    "total_assets",
    "current_assets",
    "current_liabilities",
    "cash_and_cash_equivalents",
    "total_debt",
    "stockholders_equity",
    "shares_outstanding",
}


# ---------------------------------------------------------------------------
# XBRL normalization
# ---------------------------------------------------------------------------

def normalize_xbrl_facts(
    company_facts: Optional[dict],
    max_periods: int = 12,
) -> dict[str, list[FinancialMetric]]:
    """
    Return:
        {
            metric_name: [FinancialMetric, ...]
        }

    Entries are sorted most-recent-end-date first.

    More than eight observations are retained by default because SEC Company
    Facts often contains annual, quarterly, six-month YTD and nine-month YTD
    observations. A wider history improves the chance of finding a genuinely
    comparable prior-year period.
    """
    if not company_facts:
        return {}

    us_gaap = company_facts.get("facts", {}).get("us-gaap", {})
    entity_name = company_facts.get("entityName", "")

    normalized: dict[str, list[FinancialMetric]] = {}

    for metric, concepts in METRIC_MAP.items():
        best_entries = _select_best_entries(
            us_gaap=us_gaap,
            concepts=concepts,
            is_instant=metric in INSTANT_METRICS,
            max_periods=max_periods,
        )

        if best_entries:
            normalized[metric] = best_entries

    if not normalized:
        logger.info(
            "No recognizable US-GAAP concepts found for %s",
            entity_name,
        )

    return normalized


def _select_best_entries(
    us_gaap: dict,
    concepts: list[str],
    is_instant: bool,
    max_periods: int,
) -> list[FinancialMetric]:
    """
    Select usable SEC Company Facts observations for the first preferred
    US-GAAP concept that contains data.

    Observations are deduplicated by reporting period, retaining the most
    recently filed observation for that period.
    """
    for concept in concepts:
        node = us_gaap.get(concept)

        if not node:
            continue

        units = node.get("units", {})

        raw_entries = (
            units.get("USD")
            or units.get("USD/shares")
            or units.get("shares")
            or []
        )

        if not raw_entries:
            continue

        by_period: dict[tuple, dict] = {}

        for entry in raw_entries:
            if entry.get("form") not in ("10-K", "10-Q"):
                continue

            if is_instant:
                # Instant facts should not contain a start date.
                if entry.get("start"):
                    continue

                period_key = (
                    "instant",
                    entry.get("end"),
                )

            else:
                # Duration facts require both start and end dates.
                if not entry.get("start") or not entry.get("end"):
                    continue

                period_key = (
                    entry.get("start"),
                    entry.get("end"),
                )

            existing = by_period.get(period_key)

            if (
                existing is None
                or entry.get("filed", "") > existing.get("filed", "")
            ):
                by_period[period_key] = entry

        entries = list(by_period.values())

        entries.sort(
            key=lambda entry: entry.get("end") or "",
            reverse=True,
        )

        entries = entries[:max_periods]

        if not entries:
            continue

        if "USD" in units:
            unit = "USD"
        elif "USD/shares" in units:
            unit = "USD/shares"
        else:
            unit = "shares"

        return [
            FinancialMetric(
                metric=concept,
                value=entry.get("val"),
                unit=unit,
                period=_period_label(entry),
                fiscal_period=entry.get("fp"),
                start_date=entry.get("start"),
                end_date=entry.get("end"),
                source="SEC EDGAR",
                filing_type=entry.get("form"),
                filing_date=entry.get("filed"),
                accession_number=entry.get("accn"),
            )
            for entry in entries
        ]

    return []


def _period_label(entry: dict) -> str:
    fy = entry.get("fy")
    fp = entry.get("fp", "")

    if fy and fp:
        return f"FY{fy} {fp}"

    return entry.get("end", "unknown period")


# ---------------------------------------------------------------------------
# Date and period helpers
# ---------------------------------------------------------------------------

def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _duration_days(metric: FinancialMetric) -> Optional[int]:
    """
    Return the duration of a duration-based XBRL fact.

    Instant balance-sheet facts return None.
    """
    start = _parse_date(metric.start_date)
    end = _parse_date(metric.end_date)

    if not start or not end:
        return None

    return (end - start).days


def _classify_period(metric: FinancialMetric) -> str:
    """
    Classify a duration fact using its actual date span.

    SEC fp values such as Q2 or Q3 are not sufficient by themselves because
    an observation can represent either a standalone quarter or cumulative
    year-to-date data.
    """
    days = _duration_days(metric)

    if days is None:
        return "instant"

    if 75 <= days <= 105:
        return "quarterly"

    if 150 <= days <= 210:
        return "six_month_ytd"

    if 240 <= days <= 300:
        return "nine_month_ytd"

    if 330 <= days <= 385:
        return "annual"

    return "other"


def _period_metadata(
    metric: Optional[FinancialMetric],
) -> Optional[dict]:
    """
    Produce JSON-friendly reporting-period metadata for downstream grounding.
    """
    if metric is None:
        return None

    return {
        "period_label": metric.period,
        "period_type": _classify_period(metric),
        "fiscal_period": metric.fiscal_period,
        "start_date": metric.start_date,
        "end_date": metric.end_date,
        "filing_type": metric.filing_type,
        "filing_date": metric.filing_date,
        "accession_number": metric.accession_number,
    }


def _periods_are_comparable(
    current: FinancialMetric,
    prior: FinancialMetric,
) -> bool:
    """
    Return True only when two duration facts are suitable for a YoY
    comparison.

    Requirements:
    - same period classification;
    - similar duration;
    - end dates approximately one year apart.

    This prevents invalid comparisons such as:
    - six-month YTD vs full fiscal year;
    - standalone quarter vs six-month YTD;
    - Q2 YTD vs Q1 standalone;
    - latest observation vs simply the second-latest observation.
    """
    current_type = _classify_period(current)
    prior_type = _classify_period(prior)

    valid_types = {
        "quarterly",
        "six_month_ytd",
        "nine_month_ytd",
        "annual",
    }

    if current_type not in valid_types:
        return False

    if prior_type != current_type:
        return False

    current_days = _duration_days(current)
    prior_days = _duration_days(prior)

    if current_days is None or prior_days is None:
        return False

    # Annual periods can vary because of 52/53-week fiscal calendars.
    tolerance = 14 if current_type == "annual" else 10

    if abs(current_days - prior_days) > tolerance:
        return False

    current_end = _parse_date(current.end_date)
    prior_end = _parse_date(prior.end_date)

    if not current_end or not prior_end:
        return False

    year_gap_days = (current_end - prior_end).days

    # Allows normal fiscal-calendar movement while excluding sequential periods.
    if not 340 <= year_gap_days <= 390:
        return False

    return True


def _find_comparable_prior(
    current: Optional[FinancialMetric],
    candidates: list[FinancialMetric],
) -> Optional[FinancialMetric]:
    """
    Find the most appropriate comparable prior-year observation.
    """
    if current is None:
        return None

    comparable = [
        candidate
        for candidate in candidates
        if candidate is not current
        and _periods_are_comparable(current, candidate)
    ]

    if not comparable:
        return None

    # Choose the candidate whose end date is closest to exactly one year
    # before the current period.
    current_end = _parse_date(current.end_date)

    if current_end is None:
        return None

    return min(
        comparable,
        key=lambda candidate: abs(
            (
                current_end
                - (_parse_date(candidate.end_date) or date.min)
            ).days
            - 365
        ),
    )


# ---------------------------------------------------------------------------
# Metric-selection helpers
# ---------------------------------------------------------------------------

def _latest_metric(
    normalized: dict[str, list[FinancialMetric]],
    metric: str,
) -> Optional[FinancialMetric]:
    entries = normalized.get(metric)

    if not entries:
        return None

    return entries[0]


def _latest_value(
    normalized: dict[str, list[FinancialMetric]],
    metric: str,
) -> Optional[float]:
    entry = _latest_metric(normalized, metric)

    if entry is None:
        return None

    return entry.value


def _find_matching_duration_metric(
    normalized: dict[str, list[FinancialMetric]],
    metric_name: str,
    reference: Optional[FinancialMetric],
) -> Optional[FinancialMetric]:
    """
    Find another duration metric covering the same reporting interval as the
    reference metric.

    This is used for margins and free cash flow so that, for example, annual
    revenue is never divided into six-month operating cash flow.
    """
    if reference is None:
        return None

    candidates = normalized.get(metric_name, [])

    for candidate in candidates:
        if (
            candidate.start_date == reference.start_date
            and candidate.end_date == reference.end_date
        ):
            return candidate

    return None


def _find_nearest_instant_metric(
    normalized: dict[str, list[FinancialMetric]],
    metric_name: str,
    reference_end_date: Optional[str],
) -> Optional[FinancialMetric]:
    """
    Find a point-in-time balance matching the reference period end date.

    If an exact date is unavailable, no ratio is calculated. This is more
    conservative than silently mixing different balance-sheet dates.
    """
    if not reference_end_date:
        return None

    candidates = normalized.get(metric_name, [])

    for candidate in candidates:
        if candidate.end_date == reference_end_date:
            return candidate

    return None


# ---------------------------------------------------------------------------
# Safe arithmetic
# ---------------------------------------------------------------------------

def _safe_div(
    numerator: Optional[float],
    denominator: Optional[float],
) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None

    return round(numerator / denominator, 4)


def _pct_change(
    current: Optional[float],
    prior: Optional[float],
) -> Optional[float]:
    if current is None or prior in (None, 0):
        return None

    return round(
        (current - prior) / abs(prior) * 100,
        2,
    )


def _pct(ratio: Optional[float]) -> Optional[float]:
    if ratio is None:
        return None

    return round(ratio * 100, 2)


# ---------------------------------------------------------------------------
# Deterministic financial analytics
# ---------------------------------------------------------------------------

def calculate_ratios(
    normalized: dict[str, list[FinancialMetric]],
) -> dict:
    """
    Compute deterministic ratios using compatible reporting periods.

    Important rules:
    - Growth uses comparable prior-year periods only.
    - Duration-based ratios use metrics covering the same start/end dates.
    - Balance-sheet ratios use values from the same end date.
    - Missing or non-comparable data returns None.
    - No value is fabricated, annualized, or treated as zero.
    """

    # -----------------------------------------------------------------------
    # Revenue is used as the primary reference duration for income-statement
    # margins.
    # -----------------------------------------------------------------------

    revenue_metric = _latest_metric(
        normalized,
        "revenue",
    )

    revenue = (
        revenue_metric.value
        if revenue_metric is not None
        else None
    )

    revenue_prior_metric = _find_comparable_prior(
        revenue_metric,
        normalized.get("revenue", []),
    )

    revenue_prior = (
        revenue_prior_metric.value
        if revenue_prior_metric is not None
        else None
    )

    # -----------------------------------------------------------------------
    # Net income growth uses its own comparable prior-year period.
    # -----------------------------------------------------------------------

    net_income_metric = _latest_metric(
        normalized,
        "net_income",
    )

    net_income = (
        net_income_metric.value
        if net_income_metric is not None
        else None
    )

    net_income_prior_metric = _find_comparable_prior(
        net_income_metric,
        normalized.get("net_income", []),
    )

    net_income_prior = (
        net_income_prior_metric.value
        if net_income_prior_metric is not None
        else None
    )

    # -----------------------------------------------------------------------
    # Match income-statement metrics to the exact revenue reporting interval.
    # -----------------------------------------------------------------------

    gross_profit_metric = _find_matching_duration_metric(
        normalized,
        "gross_profit",
        revenue_metric,
    )

    operating_income_metric = _find_matching_duration_metric(
        normalized,
        "operating_income",
        revenue_metric,
    )

    net_income_for_margin_metric = _find_matching_duration_metric(
        normalized,
        "net_income",
        revenue_metric,
    )

    gross_profit = (
        gross_profit_metric.value
        if gross_profit_metric is not None
        else None
    )

    operating_income = (
        operating_income_metric.value
        if operating_income_metric is not None
        else None
    )

    net_income_for_margin = (
        net_income_for_margin_metric.value
        if net_income_for_margin_metric is not None
        else None
    )

    # -----------------------------------------------------------------------
    # Balance-sheet ratios use one common latest balance-sheet date.
    # -----------------------------------------------------------------------

    total_assets_metric = _latest_metric(
        normalized,
        "total_assets",
    )

    balance_sheet_date = (
        total_assets_metric.end_date
        if total_assets_metric is not None
        else None
    )

    current_assets_metric = _find_nearest_instant_metric(
        normalized,
        "current_assets",
        balance_sheet_date,
    )

    current_liabilities_metric = _find_nearest_instant_metric(
        normalized,
        "current_liabilities",
        balance_sheet_date,
    )

    total_debt_metric = _find_nearest_instant_metric(
        normalized,
        "total_debt",
        balance_sheet_date,
    )

    equity_metric = _find_nearest_instant_metric(
        normalized,
        "stockholders_equity",
        balance_sheet_date,
    )

    total_assets = (
        total_assets_metric.value
        if total_assets_metric is not None
        else None
    )

    current_assets = (
        current_assets_metric.value
        if current_assets_metric is not None
        else None
    )

    current_liabilities = (
        current_liabilities_metric.value
        if current_liabilities_metric is not None
        else None
    )

    total_debt = (
        total_debt_metric.value
        if total_debt_metric is not None
        else None
    )

    equity = (
        equity_metric.value
        if equity_metric is not None
        else None
    )

    # -----------------------------------------------------------------------
    # ROA and ROE use net income only when its period ends on the same date as
    # the selected balance sheet. This remains a simple point-in-time ratio,
    # not an average-assets/equity calculation, so metadata explicitly records
    # that limitation.
    # -----------------------------------------------------------------------

    net_income_for_returns = None

    if (
        net_income_metric is not None
        and balance_sheet_date is not None
        and net_income_metric.end_date == balance_sheet_date
    ):
        net_income_for_returns = net_income_metric.value

    # -----------------------------------------------------------------------
    # Cash flow: use the latest operating-cash-flow period as the reference.
    # CapEx must cover exactly the same start/end dates.
    # -----------------------------------------------------------------------

    ocf_metric = _latest_metric(
        normalized,
        "operating_cash_flow",
    )

    capex_metric = _find_matching_duration_metric(
        normalized,
        "capital_expenditure",
        ocf_metric,
    )

    ocf = (
        ocf_metric.value
        if ocf_metric is not None
        else None
    )

    capex = (
        capex_metric.value
        if capex_metric is not None
        else None
    )

    free_cash_flow = None

    if ocf is not None and capex is not None:
        # SEC payment concepts normally report CapEx as a positive outflow.
        free_cash_flow = round(
            ocf - abs(capex),
            2,
        )

    # Revenue for cash-flow margins must cover the same period as OCF.
    cash_flow_revenue_metric = _find_matching_duration_metric(
        normalized,
        "revenue",
        ocf_metric,
    )

    cash_flow_revenue = (
        cash_flow_revenue_metric.value
        if cash_flow_revenue_metric is not None
        else None
    )

    # -----------------------------------------------------------------------
    # Final deterministic outputs
    # -----------------------------------------------------------------------

    ratios = {
        # Growth
        "revenue_growth_pct": _pct_change(
            revenue,
            revenue_prior,
        ),
        "net_income_growth_pct": _pct_change(
            net_income,
            net_income_prior,
        ),

        # Profitability
        "gross_margin_pct": _pct(
            _safe_div(
                gross_profit,
                revenue,
            )
        ),
        "operating_margin_pct": _pct(
            _safe_div(
                operating_income,
                revenue,
            )
        ),
        "net_margin_pct": _pct(
            _safe_div(
                net_income_for_margin,
                revenue,
            )
        ),

        # Liquidity
        "current_ratio": _safe_div(
            current_assets,
            current_liabilities,
        ),

        # Leverage
        "debt_to_equity": _safe_div(
            total_debt,
            equity,
        ),

        # Returns
        "return_on_assets_pct": _pct(
            _safe_div(
                net_income_for_returns,
                total_assets,
            )
        ),
        "return_on_equity_pct": _pct(
            _safe_div(
                net_income_for_returns,
                equity,
            )
        ),

        # Cash flow
        "free_cash_flow": free_cash_flow,
        "free_cash_flow_margin_pct": (
            _pct(
                _safe_div(
                    free_cash_flow,
                    cash_flow_revenue,
                )
            )
            if free_cash_flow is not None
            else None
        ),
        "operating_cash_flow_margin_pct": _pct(
            _safe_div(
                ocf,
                cash_flow_revenue,
            )
        ),

        # ---------------------------------------------------------------
        # Period metadata for downstream LLM grounding.
        # ---------------------------------------------------------------

        "revenue_period": _period_metadata(
            revenue_metric,
        ),

        "revenue_growth_comparison": {
            "current": _period_metadata(
                revenue_metric,
            ),
            "prior": _period_metadata(
                revenue_prior_metric,
            ),
            "comparison_type": (
                f"{_classify_period(revenue_metric)}_yoy"
                if (
                    revenue_metric is not None
                    and revenue_prior_metric is not None
                )
                else None
            ),
        },

        "net_income_period": _period_metadata(
            net_income_metric,
        ),

        "net_income_growth_comparison": {
            "current": _period_metadata(
                net_income_metric,
            ),
            "prior": _period_metadata(
                net_income_prior_metric,
            ),
            "comparison_type": (
                f"{_classify_period(net_income_metric)}_yoy"
                if (
                    net_income_metric is not None
                    and net_income_prior_metric is not None
                )
                else None
            ),
        },

        "balance_sheet_period": {
            "end_date": balance_sheet_date,
            "filing_type": (
                total_assets_metric.filing_type
                if total_assets_metric is not None
                else None
            ),
            "filing_date": (
                total_assets_metric.filing_date
                if total_assets_metric is not None
                else None
            ),
        },

        "cash_flow_period": _period_metadata(
            ocf_metric,
        ),

        "capital_expenditure_period": _period_metadata(
            capex_metric,
        ),

        "ratio_methodology_notes": {
            "growth": (
                "Growth is calculated only between comparable prior-year "
                "periods with the same duration classification."
            ),
            "margins": (
                "Income-statement margins use metrics covering the same "
                "start and end dates as revenue."
            ),
            "cash_flow": (
                "Free cash flow is calculated only when operating cash flow "
                "and capital expenditure cover the same reporting interval."
            ),
            "returns": (
                "ROA and ROE use period-end assets/equity rather than average "
                "assets/equity and should be interpreted with that limitation."
            ),
        },
    }

    return ratios