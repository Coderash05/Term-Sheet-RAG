"""Curated exemplar bank for clause family classification.

Contains 104 representative clause exemplars (8 per family across all 13 ClauseFamily values)
extracted and adapted from NVCA Model Legal Documents, Y Combinator SAFEs, and LEDGAR
out-of-domain contract provisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from termsheet_ai.schemas.contracts import ClauseFamily


@dataclass(frozen=True)
class Exemplar:
    """A curated term sheet clause exemplar used for few-shot classification and retrieval."""

    text: str
    family: ClauseFamily
    source: str
    section_header: Optional[str] = None
    notes: Optional[str] = None


def prepare_clause_text(raw_text: str, section_header: Optional[str] = None) -> str:
    """Fuse section header with raw clause text to boost classification context.

    If a section header is present and not already duplicated at the start of
    the raw text, it prepends the header in a standardized format.

    Args:
        raw_text: The verbatim clause text.
        section_header: Optional heading or numbered prefix.

    Returns:
        Prepared text with header context if applicable.
    """
    clean_text = raw_text.strip()
    if not clean_text:
        return ""

    if not section_header:
        return clean_text

    clean_header = section_header.strip().rstrip(":")
    if not clean_header:
        return clean_text

    # Check if raw_text already begins with the header (case-insensitive)
    escaped_header = re.escape(clean_header)
    if re.match(rf"^{escaped_header}\b", clean_text, flags=re.IGNORECASE):
        return clean_text

    return f"{clean_header}: {clean_text}"


# ============================================================================
# Exemplar Bank Definition (8 exemplars per family = 104 total)
# ============================================================================

_EXEMPLAR_BANK: List[Exemplar] = [
    # ------------------------------------------------------------------------
    # 1. LIQUIDATION_PREFERENCE (8 exemplars)
    # ------------------------------------------------------------------------
    Exemplar(
        text=(
            "In the event of any liquidation, dissolution or winding up of the Company, the "
            "proceeds shall be distributed first to the holders of the Series A Preferred, "
            "an amount per share equal to 1.0x the Original Issue Price, plus any declared and unpaid "
            "dividends. Thereafter, any remaining proceeds will be distributed pro-rata to the "
            "holders of Common Stock."
        ),
        family=ClauseFamily.LIQUIDATION_PREFERENCE,
        source="NVCA Model Term Sheet (2020) §3",
        section_header="Liquidation Preference",
        notes="Standard 1.0x Non-Participating Preferred",
    ),
    Exemplar(
        text=(
            "First pay 2.0x the Original Purchase Price on each share of Series A Preferred Stock. "
            "Thereafter, the remaining assets shall be distributed pro-rata to the holders of Common "
            "Stock and Preferred Stock on an as-converted to Common Stock basis until all proceeds are depleted."
        ),
        family=ClauseFamily.LIQUIDATION_PREFERENCE,
        source="NVCA Enhanced Model Term Sheet §3",
        section_header="Liquidation Preference",
        notes="2.0x Fully Participating Preferred",
    ),
    Exemplar(
        text=(
            "Upon a Liquidation Event, holders of Series A Preferred Stock receive 1.0x Original Purchase "
            "Price plus declared dividends, and shall participate pro-rata with Common Stock until "
            "they have received an aggregate return equal to 3.0x Original Purchase Price (the Cap)."
        ),
        family=ClauseFamily.LIQUIDATION_PREFERENCE,
        source="NVCA Model Term Sheet §3",
        section_header="Liquidation Preference",
        notes="1.0x Participating with 3.0x Cap",
    ),
    Exemplar(
        text=(
            "In a Deemed Liquidation Event, Series A Preferred shall rank senior in distribution "
            "priority to Series Seed Preferred and Common Stock, receiving its full preference "
            "amount prior to any junior series distribution."
        ),
        family=ClauseFamily.LIQUIDATION_PREFERENCE,
        source="NVCA Model Legal Documents",
        section_header="Seniority and Distribution",
        notes="Senior Liquidation Preference Tier",
    ),
    Exemplar(
        text=(
            "The Series A Preferred and Series Seed Preferred shall rank pari passu in liquidation "
            "preference based upon their respective aggregate original investment amounts."
        ),
        family=ClauseFamily.LIQUIDATION_PREFERENCE,
        source="NVCA Model Legal Documents",
        section_header="Pari Passu Liquidation",
        notes="Pari Passu Seniority Tier",
    ),
    Exemplar(
        text=(
            "In the event of any Liquidity Event or Dissolution Event, this Safe is entitled to "
            "receive a portion of proceeds equal to the Purchase Amount or the amount payable had "
            "the Safe converted into Common Stock immediately prior to the event."
        ),
        family=ClauseFamily.LIQUIDATION_PREFERENCE,
        source="YC Post-Money SAFE (2020) §1(b)",
        section_header="Liquidity Priority",
        notes="YC SAFE Liquidation Priority",
    ),
    Exemplar(
        text=(
            "Holders of Series B Preferred Stock shall be entitled to an upfront 3.0x non-participating "
            "liquidation preference before any distribution to Series A or Common holders."
        ),
        family=ClauseFamily.LIQUIDATION_PREFERENCE,
        source="Empirical VC Deal Terms",
        section_header="Liquidation Preference",
        notes="High Multiple (3.0x)",
    ),
    Exemplar(
        text=(
            "A Deemed Liquidation Event shall include a merger, acquisition, sale of substantially "
            "all assets, or exclusive license of all company intellectual property."
        ),
        family=ClauseFamily.LIQUIDATION_PREFERENCE,
        source="NVCA Model Term Sheet §3",
        section_header="Deemed Liquidation Events",
        notes="Deemed Liquidation Definition",
    ),
    # ------------------------------------------------------------------------
    # 2. ANTI_DILUTION (8 exemplars)
    # ------------------------------------------------------------------------
    Exemplar(
        text=(
            "The conversion price of the Series A Preferred Stock shall be subject to broad-based "
            "weighted average anti-dilution adjustment in the event that the Company issues additional "
            "shares at a purchase price less than the applicable Series A Conversion Price."
        ),
        family=ClauseFamily.ANTI_DILUTION,
        source="NVCA Model Term Sheet §4",
        section_header="Anti-Dilution Protection",
        notes="Broad-based weighted average (Market Standard)",
    ),
    Exemplar(
        text=(
            "In the event of an issuance of shares below the Series A Conversion Price, the conversion "
            "price shall be reduced on a narrow-based weighted average formula, counting only outstanding "
            "common shares and excluding unexercised options or warrants."
        ),
        family=ClauseFamily.ANTI_DILUTION,
        source="NVCA Enhanced Model Term Sheet §4",
        section_header="Anti-Dilution",
        notes="Narrow-based weighted average",
    ),
    Exemplar(
        text=(
            "If the Company issues equity securities at an effective price per share less than the "
            "Series A Conversion Price, the Series A Conversion Price shall automatically reduce to "
            "equal the lowest price per share paid in such dilutive financing (full ratchet)."
        ),
        family=ClauseFamily.ANTI_DILUTION,
        source="Empirical VC Deal Terms (Kaplan & Stromberg)",
        section_header="Full-Ratchet Anti-Dilution",
        notes="Full ratchet (High risk to founders)",
    ),
    Exemplar(
        text=(
            "Anti-dilution adjustments shall not apply to Exempted Securities, including shares "
            "reserved under the Employee Option Pool, bank debt equipment leases, or strategic mergers."
        ),
        family=ClauseFamily.ANTI_DILUTION,
        source="NVCA Model Term Sheet §4",
        section_header="Exempted Securities",
        notes="Anti-dilution carveouts",
    ),
    Exemplar(
        text=(
            "Proportional adjustment to the Series A Conversion Price will occur automatically upon "
            "any stock split, reverse stock split, stock dividend, recapitalization, or reclassification."
        ),
        family=ClauseFamily.ANTI_DILUTION,
        source="NVCA Model Term Sheet §4",
        section_header="Structural Anti-Dilution",
        notes="Stock split and reclassification adjustments",
    ),
    Exemplar(
        text=(
            "If the Company issues securities in an Equity Financing with a price per share lower "
            "than the Safe Price, the Safe shall convert into the Safe Preferred Stock at such lower price."
        ),
        family=ClauseFamily.ANTI_DILUTION,
        source="YC Post-Money SAFE §1(a)",
        section_header="Safe Price Adjustment",
        notes="SAFE conversion price down-round protection",
    ),
    Exemplar(
        text=(
            "The calculation of the adjusted Series A Conversion Price shall follow the standard NVCA "
            "equation CP2 = CP1 * (A + B) / (A + C), where A represents fully diluted capitalization."
        ),
        family=ClauseFamily.ANTI_DILUTION,
        source="NVCA Model Legal Documents",
        section_header="Conversion Price Formula",
        notes="Explicit mathematical weighted average formula",
    ),
    Exemplar(
        text=(
            "No adjustment in the Series A Conversion Price shall be made unless the adjustment "
            "would result in a change of at least $0.01 per share."
        ),
        family=ClauseFamily.ANTI_DILUTION,
        source="NVCA Model Term Sheet §4",
        section_header="De Minimis Dilution Exception",
        notes="De minimis threshold",
    ),
    # ------------------------------------------------------------------------
    # 3. BOARD_COMPOSITION (8 exemplars)
    # ------------------------------------------------------------------------
    Exemplar(
        text=(
            "The Board of Directors shall consist of five (5) members: two (2) designated by the "
            "holders of Series A Preferred, two (2) designated by the holders of Common Stock (Founders), "
            "and one (1) industry independent mutually agreed upon by Common and Preferred."
        ),
        family=ClauseFamily.BOARD_COMPOSITION,
        source="NVCA Model Term Sheet §5",
        section_header="Board of Directors",
        notes="Standard 2-2-1 board composition",
    ),
    Exemplar(
        text=(
            "The Board shall be comprised of three (3) directors: two (2) elected by the founders "
            "holding Common Stock, and one (1) elected by the lead investor holding Series A Preferred."
        ),
        family=ClauseFamily.BOARD_COMPOSITION,
        source="NVCA Model Term Sheet §5",
        section_header="Board Composition",
        notes="Founder-majority board (2 founder, 1 investor)",
    ),
    Exemplar(
        text=(
            "The Board will consist of three (3) members: two (2) designated by the Series A Investors, "
            "and one (1) designated by the Common Stockholders, providing Investors with direct voting control."
        ),
        family=ClauseFamily.BOARD_COMPOSITION,
        source="Empirical VC Deal Terms",
        section_header="Board Control",
        notes="Investor-controlled board (High risk)",
    ),
    Exemplar(
        text=(
            "The lead investor shall be entitled to designate one representative to attend all meetings "
            "of the Board of Directors in a non-voting observer capacity and receive all board packages."
        ),
        family=ClauseFamily.BOARD_COMPOSITION,
        source="NVCA Model Term Sheet §5",
        section_header="Board Observer Rights",
        notes="Non-voting observer seat",
    ),
    Exemplar(
        text=(
            "The size of the Board of Directors may not be increased or decreased without the prior "
            "affirmative vote of a majority of the outstanding Series A Preferred Stock."
        ),
        family=ClauseFamily.BOARD_COMPOSITION,
        source="NVCA Model Legal Documents",
        section_header="Board Size Changes",
        notes="Restriction on changing board size",
    ),
    Exemplar(
        text=(
            "Any director designated by Series A holders may only be removed with or without cause "
            "by the affirmative vote of the Series A holders entitled to elect such director."
        ),
        family=ClauseFamily.BOARD_COMPOSITION,
        source="NVCA Model Legal Documents",
        section_header="Director Removal and Vacancy",
        notes="Removal of investor director",
    ),
    Exemplar(
        text=(
            "The Company shall reimburse non-employee directors for reasonable out-of-pocket travel "
            "expenses incurred in connection with attending board and committee meetings."
        ),
        family=ClauseFamily.BOARD_COMPOSITION,
        source="NVCA Model Term Sheet §5",
        section_header="Board Expense Reimbursement",
        notes="Director expense reimbursement",
    ),
    Exemplar(
        text=(
            "The Board shall maintain an Audit Committee and a Compensation Committee, each of which "
            "shall include at least one director designated by the Series A Preferred."
        ),
        family=ClauseFamily.BOARD_COMPOSITION,
        source="NVCA Model Legal Documents",
        section_header="Board Committees",
        notes="Committee composition requirements",
    ),
    # ------------------------------------------------------------------------
    # 4. PROTECTIVE_PROVISIONS (8 exemplars)
    # ------------------------------------------------------------------------
    Exemplar(
        text=(
            "For so long as any shares of Series A Preferred remain outstanding, the Company shall "
            "not, without the vote or written consent of at least a majority of the Series A Preferred, "
            "alter or change the rights, preferences or privileges of the Series A Preferred."
        ),
        family=ClauseFamily.PROTECTIVE_PROVISIONS,
        source="NVCA Model Term Sheet §6",
        section_header="Protective Provisions",
        notes="Standard class veto on adverse amendments",
    ),
    Exemplar(
        text=(
            "The Company shall not effect a sale, merger, consolidation, dissolution, or liquidation "
            "of the Company without the prior written approval of a majority of the Preferred Stock."
        ),
        family=ClauseFamily.PROTECTIVE_PROVISIONS,
        source="NVCA Model Term Sheet §6",
        section_header="Protective Provisions",
        notes="Veto on M&A or Liquidation",
    ),
    Exemplar(
        text=(
            "Consent of at least 60% of the Preferred Stock is required to create, authorize, or "
            "issue any new class or series of stock having rights senior to or pari passu with Series A."
        ),
        family=ClauseFamily.PROTECTIVE_PROVISIONS,
        source="NVCA Model Term Sheet §6",
        section_header="Negative Covenants",
        notes="Veto on senior security authorization",
    ),
    Exemplar(
        text=(
            "The Company will not declare, pay, or set aside any dividend or distribution on Common "
            "Stock or repurchase any outstanding equity without separate Preferred majority approval."
        ),
        family=ClauseFamily.PROTECTIVE_PROVISIONS,
        source="NVCA Model Legal Documents",
        section_header="Dividend Restrictions",
        notes="Veto on dividends and share repurchases",
    ),
    Exemplar(
        text=(
            "Preferred majority approval shall be required to incur aggregate debt or indebtedness "
            "in excess of $250,000, outside of ordinary trade payables in the ordinary course of business."
        ),
        family=ClauseFamily.PROTECTIVE_PROVISIONS,
        source="NVCA Enhanced Model Term Sheet §6",
        section_header="Debt Incurrence Limitations",
        notes="Debt ceiling covenant",
    ),
    Exemplar(
        text=(
            "The Company shall not materially alter or change the primary business plan or line of "
            "business without the prior consent of the Series A Director."
        ),
        family=ClauseFamily.PROTECTIVE_PROVISIONS,
        source="NVCA Enhanced Model Term Sheet §6",
        section_header="Business Line Restriction",
        notes="Operational veto over corporate strategy",
    ),
    Exemplar(
        text=(
            "Preferred consent is mandatory to increase or decrease the authorized number of shares "
            "of Common Stock, Preferred Stock, or shares reserved under the equity incentive plan."
        ),
        family=ClauseFamily.PROTECTIVE_PROVISIONS,
        source="NVCA Model Term Sheet §6",
        section_header="Capital Stock Changes",
        notes="Capital stock authorization veto",
    ),
    Exemplar(
        text=(
            "The Company shall not enter into any transaction with any founder, officer, or affiliate "
            "on terms less favorable than arm's length without approval of the Preferred director."
        ),
        family=ClauseFamily.PROTECTIVE_PROVISIONS,
        source="NVCA Model Term Sheet §6",
        section_header="Affiliate Transactions",
        notes="Interested-party transaction veto",
    ),
    # ------------------------------------------------------------------------
    # 5. FOUNDER_VESTING (8 exemplars)
    # ------------------------------------------------------------------------
    Exemplar(
        text=(
            "All founder shares shall be subject to a 4-year vesting schedule with a 1-year cliff. "
            "25% of shares shall vest on the first anniversary of the vesting commencement date, and "
            "the remaining 75% shall vest in equal monthly installments over the following 36 months."
        ),
        family=ClauseFamily.FOUNDER_VESTING,
        source="NVCA Model Term Sheet §7",
        section_header="Founder Vesting",
        notes="Standard 4-year vesting with 1-year cliff",
    ),
    Exemplar(
        text=(
            "Upon a Change of Control of the Company, if a founder is terminated without Cause or "
            "resigns for Good Reason within 12 months, 100% of their unvested shares shall accelerate (double trigger)."
        ),
        family=ClauseFamily.FOUNDER_VESTING,
        source="NVCA Model Term Sheet §7",
        section_header="Vesting Acceleration",
        notes="Double-trigger acceleration (Market Standard)",
    ),
    Exemplar(
        text=(
            "In the event of an acquisition of the Company, 50% of each founder's remaining unvested "
            "shares shall accelerate immediately upon closing of the transaction (single trigger)."
        ),
        family=ClauseFamily.FOUNDER_VESTING,
        source="Empirical VC Deal Terms",
        section_header="Single-Trigger Acceleration",
        notes="Single-trigger acceleration",
    ),
    Exemplar(
        text=(
            "The Company shall retain an option to repurchase any unvested shares from a departing founder "
            "at the original purchase price upon cessation of service as an employee or consultant."
        ),
        family=ClauseFamily.FOUNDER_VESTING,
        source="NVCA Model Legal Documents",
        section_header="Repurchase Option",
        notes="Repurchase of unvested equity",
    ),
    Exemplar(
        text=(
            "All existing shares held by founders prior to the Series A round shall restart vesting "
            "from day one under a newly imposed 4-year vesting schedule, with zero credit for prior service."
        ),
        family=ClauseFamily.FOUNDER_VESTING,
        source="Empirical VC Deal Terms",
        section_header="Vesting Reset",
        notes="Founder vesting reset (Aggressive term)",
    ),
    Exemplar(
        text=(
            "Vesting shall cease immediately upon voluntary resignation or termination of employment "
            "for Cause, and no further shares shall vest following the date of termination."
        ),
        family=ClauseFamily.FOUNDER_VESTING,
        source="NVCA Model Legal Documents",
        section_header="Cessation of Vesting",
        notes="Termination of employment effect on vesting",
    ),
    Exemplar(
        text=(
            "Founders shall timely file a valid Section 83(b) election with the Internal Revenue Service "
            "within 30 days of the stock purchase agreement or receipt of restricted shares."
        ),
        family=ClauseFamily.FOUNDER_VESTING,
        source="NVCA Model Term Sheet §7",
        section_header="Section 83(b) Election",
        notes="83(b) tax covenant",
    ),
    Exemplar(
        text=(
            "Founders may transfer vested restricted shares to immediate family trusts for estate planning, "
            "provided transferees agree in writing to be bound by the vesting agreement."
        ),
        family=ClauseFamily.FOUNDER_VESTING,
        source="NVCA Model Legal Documents",
        section_header="Permitted Transfers of Vested Shares",
        notes="Estate planning transfers",
    ),
    # ------------------------------------------------------------------------
    # 6. INFORMATION_RIGHTS (8 exemplars)
    # ------------------------------------------------------------------------
    Exemplar(
        text=(
            "The Company shall deliver to each Major Investor annual audited financial statements "
            "within 90 days after the end of each fiscal year, prepared in accordance with GAAP."
        ),
        family=ClauseFamily.INFORMATION_RIGHTS,
        source="NVCA Model Term Sheet §8",
        section_header="Information Rights",
        notes="Annual audited financials covenant",
    ),
    Exemplar(
        text=(
            "The Company shall furnish monthly financial statements, including balance sheet and "
            "cash flow statement, to Major Investors within 30 days after the end of each month."
        ),
        family=ClauseFamily.INFORMATION_RIGHTS,
        source="NVCA Model Term Sheet §8",
        section_header="Financial Reporting",
        notes="Monthly financial reporting",
    ),
    Exemplar(
        text=(
            "The Company will provide an annual comprehensive operating budget and business plan "
            "to Major Investors at least 30 days prior to the beginning of each fiscal year."
        ),
        family=ClauseFamily.INFORMATION_RIGHTS,
        source="NVCA Model Term Sheet §8",
        section_header="Budget and Operating Plan",
        notes="Annual operating budget delivery",
    ),
    Exemplar(
        text=(
            "Major Investors shall have the right to visit and inspect the properties of the Company, "
            "examine its books and corporate records, and discuss its affairs with officers during normal business hours."
        ),
        family=ClauseFamily.INFORMATION_RIGHTS,
        source="NVCA Model Legal Documents",
        section_header="Inspection Rights",
        notes="Books and records inspection rights",
    ),
    Exemplar(
        text=(
            "A 'Major Investor' entitled to information and inspection rights shall be defined as any "
            "holder of at least 500,000 shares of Series A Preferred Stock."
        ),
        family=ClauseFamily.INFORMATION_RIGHTS,
        source="NVCA Model Term Sheet §8",
        section_header="Major Investor Definition",
        notes="Major investor share threshold",
    ),
    Exemplar(
        text=(
            "The covenants set forth in this Section shall terminate upon the closing of an Initial "
            "Public Offering (IPO) or upon an acquisition of the Company."
        ),
        family=ClauseFamily.INFORMATION_RIGHTS,
        source="NVCA Model Term Sheet §8",
        section_header="Termination of Information Rights",
        notes="Termination upon IPO",
    ),
    Exemplar(
        text=(
            "Any investor receiving confidential company financial information agrees to maintain strict "
            "confidentiality and not disclose proprietary operational metrics to competitors."
        ),
        family=ClauseFamily.INFORMATION_RIGHTS,
        source="NVCA Model Legal Documents",
        section_header="Confidentiality of Financials",
        notes="Confidentiality covenant for reports",
    ),
    Exemplar(
        text=(
            "Safe holders will have no voting rights or information rights regarding financial statements "
            "prior to the conversion of this Safe into shares of Safe Preferred Stock."
        ),
        family=ClauseFamily.INFORMATION_RIGHTS,
        source="YC Post-Money SAFE §4",
        section_header="No Information Rights",
        notes="SAFE lack of information rights prior to conversion",
    ),
    # ------------------------------------------------------------------------
    # 7. ROFR_CO_SALE (8 exemplars)
    # ------------------------------------------------------------------------
    Exemplar(
        text=(
            "If any Founder proposes to sell or transfer shares of Common Stock, the Company shall have "
            "a Right of First Refusal (ROFR) to purchase such shares on the same terms and conditions."
        ),
        family=ClauseFamily.ROFR_CO_SALE,
        source="NVCA Model Term Sheet §9",
        section_header="Right of First Refusal",
        notes="Primary company ROFR on founder sales",
    ),
    Exemplar(
        text=(
            "To the extent the Company does not exercise its Right of First Refusal, each Major Investor "
            "shall have a secondary right to purchase its pro-rata share of the offered founder shares."
        ),
        family=ClauseFamily.ROFR_CO_SALE,
        source="NVCA Model Term Sheet §9",
        section_header="Secondary ROFR",
        notes="Secondary investor ROFR",
    ),
    Exemplar(
        text=(
            "If any founder transfers Common Stock, Major Investors shall have a Co-Sale right to "
            "participate on a pro-rata basis in the sale on the same terms and price as the founder."
        ),
        family=ClauseFamily.ROFR_CO_SALE,
        source="NVCA Model Term Sheet §9",
        section_header="Co-Sale Rights",
        notes="Co-sale / Tag-along participation right",
    ),
    Exemplar(
        text=(
            "Before any proposed transfer, the transferring shareholder must deliver a written Notice "
            "of Transfer specifying the proposed buyer, purchase price, and number of shares to be transferred."
        ),
        family=ClauseFamily.ROFR_CO_SALE,
        source="NVCA Model Legal Documents",
        section_header="Transfer Notice",
        notes="Formal notice requirement for transfer",
    ),
    Exemplar(
        text=(
            "The Right of First Refusal and Co-Sale shall not apply to Permitted Transfers, including "
            "bona fide gifts, transfers to immediate family, or transfers to revocable living trusts."
        ),
        family=ClauseFamily.ROFR_CO_SALE,
        source="NVCA Model Legal Documents",
        section_header="Permitted Transfers",
        notes="Exempt transfers under ROFR",
    ),
    Exemplar(
        text=(
            "If neither the Company nor the Investors exercise their ROFR within 30 days, the founder "
            "may consummate the proposed sale within 60 days on the exact terms in the Transfer Notice."
        ),
        family=ClauseFamily.ROFR_CO_SALE,
        source="NVCA Model Term Sheet §9",
        section_header="Expiration of Refusal Period",
        notes="Waiver of ROFR after lapse of time",
    ),
    Exemplar(
        text=(
            "The pro-rata share for purposes of ROFR and Co-Sale rights shall be calculated based upon "
            "fully diluted equity ownership of all participating preferred shareholders."
        ),
        family=ClauseFamily.ROFR_CO_SALE,
        source="NVCA Model Legal Documents",
        section_header="Pro-Rata Calculation",
        notes="Pro-rata allocation formula",
    ),
    Exemplar(
        text=(
            "No transfer of Common Stock by founders shall be valid unless the proposed transferee signs "
            "a joinder agreement agreeing to be bound by the ROFR and Co-Sale Agreement."
        ),
        family=ClauseFamily.ROFR_CO_SALE,
        source="NVCA Model Legal Documents",
        section_header="Transferee Joinder",
        notes="Condition precedent to transferee validity",
    ),
    # ------------------------------------------------------------------------
    # 8. DRAG_ALONG (8 exemplars)
    # ------------------------------------------------------------------------
    Exemplar(
        text=(
            "If the Board of Directors and holders of a majority of Preferred Stock approve an M&A "
            "sale of the Company, all stockholders shall be required to vote in favor of such transaction (Drag-Along)."
        ),
        family=ClauseFamily.DRAG_ALONG,
        source="NVCA Model Term Sheet §10",
        section_header="Drag-Along Rights",
        notes="Standard drag-along provision",
    ),
    Exemplar(
        text=(
            "The Drag-Along right shall only be exercisable if the enterprise value or net per-share "
            "consideration in the transaction equals or exceeds a minimum valuation floor of $50,000,000."
        ),
        family=ClauseFamily.DRAG_ALONG,
        source="NVCA Enhanced Model Term Sheet §10",
        section_header="Drag-Along Minimum Floor",
        notes="Valuation hurdle for drag-along",
    ),
    Exemplar(
        text=(
            "Shareholders dragged into an acquisition shall not be required to make representations "
            "beyond their own title to shares, nor enter into non-compete agreements."
        ),
        family=ClauseFamily.DRAG_ALONG,
        source="NVCA Model Term Sheet §10",
        section_header="Drag-Along Covenants",
        notes="Founder protection against uncapped indemnification",
    ),
    Exemplar(
        text=(
            "All stockholders participating in a Drag-Along sale shall receive the same form and per-share "
            "consideration as other holders of the same class of stock, subject to liquidation preferences."
        ),
        family=ClauseFamily.DRAG_ALONG,
        source="NVCA Model Term Sheet §10",
        section_header="Consideration Parity",
        notes="Equal treatment requirement under drag",
    ),
    Exemplar(
        text=(
            "Each dragged stockholder irrevocably appoints the CEO and Lead Investor as their proxy "
            "and attorney-in-fact to execute any transaction documents required to effectuate the sale."
        ),
        family=ClauseFamily.DRAG_ALONG,
        source="NVCA Model Legal Documents",
        section_header="Proxy Appointment",
        notes="Power of attorney / proxy clause",
    ),
    Exemplar(
        text=(
            "Stockholders subject to Drag-Along covenants agree to waive any statutory appraisal rights "
            "or dissenters' rights under Delaware General Corporation Law Section 262."
        ),
        family=ClauseFamily.DRAG_ALONG,
        source="NVCA Model Term Sheet §10",
        section_header="Waiver of Appraisal Rights",
        notes="Delaware appraisal rights waiver",
    ),
    Exemplar(
        text=(
            "The Drag-Along threshold shall require the affirmative vote of at least 60% of Preferred "
            "Stock and a majority of Common Stock before founders may be forced into an acquisition."
        ),
        family=ClauseFamily.DRAG_ALONG,
        source="NVCA Model Legal Documents",
        section_header="Drag Threshold",
        notes="Dual-class threshold requirement",
    ),
    Exemplar(
        text=(
            "Liability for indemnification claims under the Drag-Along agreement shall be capped at "
            "the net proceeds actually received by each individual shareholder in the transaction."
        ),
        family=ClauseFamily.DRAG_ALONG,
        source="NVCA Model Term Sheet §10",
        section_header="Indemnity Cap",
        notes="Indemnity cap protection",
    ),
    # ------------------------------------------------------------------------
    # 9. OPTION_POOL (8 exemplars)
    # ------------------------------------------------------------------------
    Exemplar(
        text=(
            "The Company's fully diluted capitalization shall include an unallocated employee option "
            "pool representing 10% of the post-financing capitalization reserved for future hires."
        ),
        family=ClauseFamily.OPTION_POOL,
        source="NVCA Model Term Sheet §11",
        section_header="Option Pool",
        notes="Standard 10% post-closing option pool",
    ),
    Exemplar(
        text=(
            "The agreed pre-money valuation shall be deemed to include an increase in the unallocated "
            "option pool to 15% on a post-closing basis, with all dilution borne solely by founders."
        ),
        family=ClauseFamily.OPTION_POOL,
        source="Empirical VC Deal Terms",
        section_header="Pre-Money Option Pool",
        notes="Pre-money option pool expansion (founder dilution)",
    ),
    Exemplar(
        text=(
            "The option pool shall be created on a post-money basis so that incoming Series A investors "
            "and existing common shareholders share the dilution proportionally."
        ),
        family=ClauseFamily.OPTION_POOL,
        source="NVCA Enhanced Model Term Sheet §11",
        section_header="Post-Money Option Pool",
        notes="Post-money option pool creation (founder-friendly)",
    ),
    Exemplar(
        text=(
            "The Company shall reserve 1,200,000 shares of Common Stock for issuance to employees, "
            "consultants, and directors under the amended 2026 Equity Incentive Plan."
        ),
        family=ClauseFamily.OPTION_POOL,
        source="NVCA Model Legal Documents",
        section_header="Incentive Plan Reservation",
        notes="Fixed share reservation for employee plan",
    ),
    Exemplar(
        text=(
            "Options granted under the Option Pool shall have an exercise price equal to fair market "
            "value as determined in good faith by the Board in compliance with Section 409A."
        ),
        family=ClauseFamily.OPTION_POOL,
        source="NVCA Model Legal Documents",
        section_header="Option Exercise Price (409A)",
        notes="409A valuation compliance for stock options",
    ),
    Exemplar(
        text=(
            "Any shares of Common Stock forfeited or repurchased from terminated employees shall "
            "automatically return to the available Option Pool for future reissuance."
        ),
        family=ClauseFamily.OPTION_POOL,
        source="NVCA Model Legal Documents",
        section_header="Forfeited Option Pool Shares",
        notes="Return of forfeited shares to pool",
    ),
    Exemplar(
        text=(
            "Any expansion or increase in the total number of shares reserved under the Option Pool "
            "shall require the prior written consent of the Series A Preferred Director."
        ),
        family=ClauseFamily.OPTION_POOL,
        source="NVCA Model Term Sheet §11",
        section_header="Option Pool Expansion Veto",
        notes="Protective approval on option pool increase",
    ),
    Exemplar(
        text=(
            "Prior to the closing of the next qualified financing, the Company shall refresh the "
            "employee option pool to maintain at least 8% unallocated equity."
        ),
        family=ClauseFamily.OPTION_POOL,
        source="NVCA Enhanced Model Term Sheet §11",
        section_header="Option Pool Refresh",
        notes="Mandatory option pool refresh covenant",
    ),
    # ------------------------------------------------------------------------
    # 10. EXCLUSIVITY_NO_SHOP (8 exemplars)
    # ------------------------------------------------------------------------
    Exemplar(
        text=(
            "For a period of 30 days following the execution of this Term Sheet, the Company and Founders "
            "agree not to solicit, encourage, initiate, or engage in any discussions regarding alternative financing."
        ),
        family=ClauseFamily.EXCLUSIVITY_NO_SHOP,
        source="NVCA Model Term Sheet §12",
        section_header="Exclusivity / No-Shop",
        notes="Standard 30-day binding exclusivity",
    ),
    Exemplar(
        text=(
            "The Company agrees to a 45-day binding no-shop covenant prohibiting any negotiations "
            "concerning any merger, sale of assets, or sale of capital stock with any third party."
        ),
        family=ClauseFamily.EXCLUSIVITY_NO_SHOP,
        source="NVCA Model Term Sheet §12",
        section_header="No-Shop Agreement",
        notes="45-day extended no-shop period",
    ),
    Exemplar(
        text=(
            "The Company shall immediately inform the Lead Investor in writing upon receipt of any "
            "unsolicited offer, inquiry, or term sheet from any competing venture capital fund."
        ),
        family=ClauseFamily.EXCLUSIVITY_NO_SHOP,
        source="NVCA Enhanced Model Term Sheet §12",
        section_header="Notice of Unsolicited Inquiries",
        notes="Notice requirement on competing bids",
    ),
    Exemplar(
        text=(
            "If the Company terminates negotiations or breaches this exclusivity covenant to accept an "
            "alternative term sheet, it shall pay investor reasonable legal expenses up to $50,000 (Break-up Fee)."
        ),
        family=ClauseFamily.EXCLUSIVITY_NO_SHOP,
        source="Empirical VC Deal Terms",
        section_header="Break-Up Fee",
        notes="Break-up fee / expense reimbursement penalty",
    ),
    Exemplar(
        text=(
            "The Exclusivity and Confidentiality provisions of this Term Sheet shall be legally binding "
            "upon the parties, notwithstanding the non-binding nature of all other terms."
        ),
        family=ClauseFamily.EXCLUSIVITY_NO_SHOP,
        source="NVCA Model Term Sheet §12",
        section_header="Binding Nature of Exclusivity",
        notes="Carve-out affirming binding status of no-shop",
    ),
    Exemplar(
        text=(
            "Exclusivity shall automatically terminate if the Lead Investor notifies the Company in "
            "writing that it is abandoning or withdrawing its financing proposal."
        ),
        family=ClauseFamily.EXCLUSIVITY_NO_SHOP,
        source="NVCA Model Legal Documents",
        section_header="Termination of Exclusivity",
        notes="Early termination upon investor withdrawal",
    ),
    Exemplar(
        text=(
            "During the exclusivity period, the Company and its legal counsel agree to negotiate in "
            "good faith to complete definitive legal documents consistent with this Term Sheet."
        ),
        family=ClauseFamily.EXCLUSIVITY_NO_SHOP,
        source="NVCA Model Term Sheet §12",
        section_header="Good Faith Negotiations",
        notes="Good faith negotiation covenant",
    ),
    Exemplar(
        text=(
            "The Company shall immediately instruct its officers, advisors, and investment bankers to "
            "cease all ongoing discussions with other potential financing sources."
        ),
        family=ClauseFamily.EXCLUSIVITY_NO_SHOP,
        source="NVCA Model Legal Documents",
        section_header="Cessation of Discussions",
        notes="Immediate termination of competing talks",
    ),
    # ------------------------------------------------------------------------
    # 11. CONVERSION_RIGHTS (8 exemplars)
    # ------------------------------------------------------------------------
    Exemplar(
        text=(
            "Each share of Series A Preferred Stock shall be convertible at any time at the option of "
            "the holder into shares of Common Stock at an initial conversion ratio of 1:1."
        ),
        family=ClauseFamily.CONVERSION_RIGHTS,
        source="NVCA Model Term Sheet §13",
        section_header="Optional Conversion",
        notes="Standard optional 1:1 conversion right",
    ),
    Exemplar(
        text=(
            "All outstanding Preferred Stock shall automatically convert into Common Stock upon the "
            "closing of an underwritten Initial Public Offering raising at least $50,000,000 (QPO)."
        ),
        family=ClauseFamily.CONVERSION_RIGHTS,
        source="NVCA Model Term Sheet §13",
        section_header="Automatic Conversion on IPO",
        notes="Qualified Public Offering automatic conversion",
    ),
    Exemplar(
        text=(
            "Automatic conversion shall occur upon the affirmative written election or consent of "
            "the holders of at least a majority of the outstanding Series A Preferred Stock."
        ),
        family=ClauseFamily.CONVERSION_RIGHTS,
        source="NVCA Model Term Sheet §13",
        section_header="Mandatory Conversion by Vote",
        notes="Majority preferred election to convert",
    ),
    Exemplar(
        text=(
            "Upon conversion, no fractional shares of Common Stock will be issued, and the Company "
            "shall pay cash in lieu of any fractional share based on fair market value."
        ),
        family=ClauseFamily.CONVERSION_RIGHTS,
        source="NVCA Model Legal Documents",
        section_header="Fractional Shares on Conversion",
        notes="Cash in lieu of fractional shares",
    ),
    Exemplar(
        text=(
            "This Safe will automatically convert into shares of Safe Preferred Stock upon the closing "
            "of an Equity Financing in which the Company issues and sells Preferred Stock."
        ),
        family=ClauseFamily.CONVERSION_RIGHTS,
        source="YC Post-Money SAFE §1(a)",
        section_header="Safe Equity Financing Conversion",
        notes="YC SAFE automatic conversion mechanism",
    ),
    Exemplar(
        text=(
            "The Company shall at all times reserve and keep available out of its authorized Common "
            "Stock a sufficient number of shares to effect the conversion of all Preferred Stock."
        ),
        family=ClauseFamily.CONVERSION_RIGHTS,
        source="NVCA Model Legal Documents",
        section_header="Reservation of Common Stock",
        notes="Underlying common share reserve covenant",
    ),
    Exemplar(
        text=(
            "In any recapitalization, merger, or reorganization, provision shall be made so holders "
            "of Preferred Stock receive the equivalent securities they would have received had they converted."
        ),
        family=ClauseFamily.CONVERSION_RIGHTS,
        source="NVCA Model Term Sheet §13",
        section_header="Conversion Adjustment on Reorganization",
        notes="Conversion right preservation in corporate reorganizations",
    ),
    Exemplar(
        text=(
            "Surrender of preferred stock certificates at the office of the transfer agent shall be "
            "accompanied by written notice stating the holder's election to convert."
        ),
        family=ClauseFamily.CONVERSION_RIGHTS,
        source="NVCA Model Legal Documents",
        section_header="Conversion Mechanics",
        notes="Procedural conversion mechanics",
    ),
    # ------------------------------------------------------------------------
    # 12. REDEMPTION_RIGHTS (8 exemplars)
    # ------------------------------------------------------------------------
    Exemplar(
        text=(
            "The Series A Preferred Stock shall be redeemable at the option of holders of a majority of "
            "the Series A Preferred at any time after the fifth (5th) anniversary of the Closing Date."
        ),
        family=ClauseFamily.REDEMPTION_RIGHTS,
        source="NVCA Model Term Sheet §14",
        section_header="Redemption Rights",
        notes="Investor-optional redemption after 5 years",
    ),
    Exemplar(
        text=(
            "Redemption shall be payable in three equal annual installments at a price per share equal "
            "to the Original Issue Price plus all accrued or declared and unpaid dividends."
        ),
        family=ClauseFamily.REDEMPTION_RIGHTS,
        source="NVCA Model Term Sheet §14",
        section_header="Redemption Installments",
        notes="3 equal annual redemption payments",
    ),
    Exemplar(
        text=(
            "If the Company does not have sufficient legally available funds to redeem all requested "
            "shares, the Company shall redeem shares pro-rata and redeem remaining shares as funds become available."
        ),
        family=ClauseFamily.REDEMPTION_RIGHTS,
        source="NVCA Model Legal Documents",
        section_header="Inability to Redeem",
        notes="Statutory solvency limitation on redemption",
    ),
    Exemplar(
        text=(
            "In the event of a default in redemption payments, the applicable redemption price shall "
            "bear interest at the rate of 12% per annum until paid in full."
        ),
        family=ClauseFamily.REDEMPTION_RIGHTS,
        source="NVCA Enhanced Model Term Sheet §14",
        section_header="Redemption Default Interest",
        notes="Default penalty interest on failure to redeem",
    ),
    Exemplar(
        text=(
            "If the Company defaults on its redemption obligations, the Series A holders shall have "
            "the right to elect a majority of the Board of Directors until the redemption is completed."
        ),
        family=ClauseFamily.REDEMPTION_RIGHTS,
        source="Empirical VC Deal Terms",
        section_header="Board Takeover on Redemption Default",
        notes="Board takeover penalty on redemption default",
    ),
    Exemplar(
        text=(
            "The Preferred Stock shall not be callable or subject to mandatory redemption by the Company "
            "without the unanimous written consent of all Series A holders."
        ),
        family=ClauseFamily.REDEMPTION_RIGHTS,
        source="NVCA Model Legal Documents",
        section_header="Prohibition on Company Call",
        notes="Company call rights prohibited",
    ),
    Exemplar(
        text=(
            "Written notice of redemption must be delivered by holders of Preferred Stock to the Company "
            "at least 60 days prior to the desired initial redemption date."
        ),
        family=ClauseFamily.REDEMPTION_RIGHTS,
        source="NVCA Model Legal Documents",
        section_header="Redemption Notice Requirements",
        notes="Notice timeline for redemption election",
    ),
    Exemplar(
        text=(
            "The Series A Preferred Stock shall have no mandatory redemption rights or sinking fund "
            "provisions, remaining permanent equity until converted or liquidated."
        ),
        family=ClauseFamily.REDEMPTION_RIGHTS,
        source="NVCA Model Term Sheet §14",
        section_header="No Redemption (Market Standard)",
        notes="Disclaimer of redemption rights (Standard VC norm)",
    ),
    # ------------------------------------------------------------------------
    # 13. OTHER (Negative / Out-of-Domain Boilerplate) (8 exemplars)
    # ------------------------------------------------------------------------
    Exemplar(
        text=(
            "This Agreement and all acts and transactions contemplated hereby shall be governed by, "
            "construed and enforced in accordance with the laws of the State of Delaware."
        ),
        family=ClauseFamily.OTHER,
        source="LEDGAR Boilerplate / NVCA Miscellaneous §15",
        section_header="Governing Law",
        notes="Delaware governing law clause",
    ),
    Exemplar(
        text=(
            "If any provision of this Agreement is held to be invalid, illegal, or unenforceable, the "
            "validity, legality, and enforceability of the remaining provisions shall not in any way be affected."
        ),
        family=ClauseFamily.OTHER,
        source="LEDGAR Boilerplate Provisions (Tuggener et al., 2020)",
        section_header="Severability",
        notes="Standard contract severability provision",
    ),
    Exemplar(
        text=(
            "This Term Sheet may be executed in two or more counterparts, each of which shall be deemed "
            "an original, but all of which together shall constitute one and the same instrument."
        ),
        family=ClauseFamily.OTHER,
        source="NVCA Model Term Sheet §15",
        section_header="Counterparts",
        notes="Execution in counterparts clause",
    ),
    Exemplar(
        text=(
            "This Agreement constitutes the entire understanding between the parties with respect to the "
            "subject matter hereof and supersedes all prior agreements, oral or written negotiations."
        ),
        family=ClauseFamily.OTHER,
        source="LEDGAR Boilerplate Provisions",
        section_header="Entire Agreement / Integration",
        notes="Integration and merger clause",
    ),
    Exemplar(
        text=(
            "All notices and other communications hereunder shall be in writing and shall be deemed to "
            "have been given when delivered personally or sent by confirmed electronic transmission."
        ),
        family=ClauseFamily.OTHER,
        source="LEDGAR Boilerplate Provisions",
        section_header="Notices",
        notes="Standard notice provision",
    ),
    Exemplar(
        text=(
            "Neither party may assign its rights or delegate its obligations under this Agreement "
            "without the prior written consent of the other party."
        ),
        family=ClauseFamily.OTHER,
        source="LEDGAR Boilerplate Provisions",
        section_header="Assignment",
        notes="Assignment restriction boilerplate",
    ),
    Exemplar(
        text=(
            "Each party shall bear its own legal, accounting, and advisory fees and out-of-pocket expenses "
            "incurred in connection with the preparation and negotiation of this Agreement."
        ),
        family=ClauseFamily.OTHER,
        source="LEDGAR / NVCA Miscellaneous",
        section_header="Expenses and Fees",
        notes="General transaction expenses boilerplate",
    ),
    Exemplar(
        text=(
            "Except for the sections titled 'Exclusivity' and 'Confidentiality', this Term Sheet is "
            "intended solely as a summary of discussion terms and does not constitute a binding legal agreement."
        ),
        family=ClauseFamily.OTHER,
        source="NVCA Model Term Sheet Preamble",
        section_header="Non-Binding Statement of Intent",
        notes="Non-binding term sheet preamble",
    ),
]


# ============================================================================
# Helper Functions
# ============================================================================


def load_exemplar_bank() -> List[Exemplar]:
    """Return the complete list of 104 curated exemplars."""
    return list(_EXEMPLAR_BANK)


def get_training_corpus() -> Tuple[List[str], List[ClauseFamily], List[str]]:
    """Return prepared texts, target families, and raw texts for model training.

    Returns:
        Tuple of (prepared_texts, families, sources).
    """
    prepared_texts: List[str] = []
    families: List[ClauseFamily] = []
    sources: List[str] = []

    for ex in _EXEMPLAR_BANK:
        prep = prepare_clause_text(ex.text, ex.section_header)
        prepared_texts.append(prep)
        families.append(ex.family)
        sources.append(ex.source)

    return prepared_texts, families, sources


def get_exemplar_counts() -> Dict[str, int]:
    """Return a dictionary mapping each clause family value to its exemplar count."""
    counts: Dict[str, int] = {f.value: 0 for f in ClauseFamily}
    for ex in _EXEMPLAR_BANK:
        counts[ex.family.value] += 1
    return counts
