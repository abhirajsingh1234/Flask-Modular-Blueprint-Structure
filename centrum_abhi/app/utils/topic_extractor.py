
def extract_from_headers_list(text,file):
 
    sections = []

    if file == 'CBL_Nomination_and_Remuneration_Policy.pdf':
        headers=['1. OBJECTIVE & APPLICABILITY ','2. DEFINITIONS','3. ROLE OF COMMITTEE ',

                '4. CONSTITUTION OF COMMITTEE:','5. NOMINATION DUTIES','6.  REMUNERATION DUTIES','7. MINUTES OF COMMITTEE MEETING',

                '8. APPLICABILITY TO SUBSIDIARIES ','9. REVIEW AND AMMENDMENT ','10. COMPLIANCE RESPONSIBILITY ']
    
    elif file =='Facility_of_Voluntary_Freezing_of_Trading_Accounts_by_Clients.pdf':
        headers=['Policy on Facility of voluntary freezing of Trading Accounts by Clients']
    
    elif file =='Inactive_TradingAccount_Policy_version3.pdf':
        headers=['POLICY ON INACTIVE TRADING ACCOUNTS ','1. Reactivation of inactive Accounts '] 

    elif file == 'InternalAuctionPolicy.pdf':
        headers=['''INTERNAL AUCTION POLICY''']

    elif file =='Internal-policy-on-NISM-VII.pdf':
        headers=['INTERNAL POLICY ON NISM-VII: Securities Operations and Risk Management Certification']

    elif file =='INVESTOR GRIEVANCES POLICY.pdf':
        headers=['INVESTOR GRIEVANCE POLICY']

    elif file =='Rights-And-Obligations-English.pdf':
        headers=['Annexure – 4 RIGHTS AND OBLIGATIONS OF STOCK BROKERS, SUB-BROKERS AND CLIENTS as prescribed by SEBI and Stock Exchanges ',
        'CLIENT INFORMATION ','MARGINS 11.','TRANSACTIONS AND SETTLEMENTS ','LIQUIDATION AND CLOSE OUT OF POSITION ',
        'DISPUTE RESOLUTION ','TERMINATION OF RELATIONSHIP','ADDITIONAL RIGHTS AND OBLIGATIONS ','ELECTRONIC CONTRACT NOTES (ECN) ','LAW AND JURISDICTION'
        ,'INTERNET & WIRELESS TECHNOLOGY BASED TRADING FACILITY PROVIDED BY STOCK BROKERS TO CLIENT (All the clauses mentioned in the ‘Rights and Obligations’ document(s) shall be applicable. Additionally, the clauses mentioned herein shall also be applicable.)  '
        ]

    elif file =='Risk_Policy.pdf':
        headers=['1. Introduction','2. Trading Limits & Other RMS Criteria','3. Margin collection and requirements',
        '4. Risk Square off policy ','5. CBL RMS Discretion in Exceptional Circumstances ','Margin Trading Funding (MTF) Risk Policy'
        'Policy on Facility of voluntary freezing of Trading Accounts by Clients']
    

    for i, header in enumerate(headers):

        try:

            start_index = text.index(header)

            end_index = text.index(headers[i + 1]) if i + 1 < len(headers) else len(text)

            content = text[start_index:end_index].strip()

            sections.append({


                "metadata": {
                    
                    'source': file,            
                    
                    "topic": header.strip(),

                    "subtopic": ''},

                "content": content

            })
            print(f"{header} — extracted.")

        except ValueError as ve:
            print(f"[!] Header not found: {header} — skipping.")

            continue

    return sections

def extract_role_of_committee_subsections(text, file):

    """
    Extracts hardcoded subsections (3.1, 3.2, 3.3) with their full headings and content
    from the '3. ROLE OF COMMITTEE' section text.

    Args:
        text (str): Full content of the '3. ROLE OF COMMITTEE' section.

    Returns:
        List[Dict]: Each dict has 'topic' and 'content' keys.
    """
    if file == 'CBL_Nomination_and_Remuneration_Policy.pdf':
        topic = '3. ROLE OF COMMITTEE'
        headers = [
            '3.1. Matters to be dealt with, perused and recommended to the Board by the Nomination',
            '3.2.   Policy for appointment and removal of Director, KMP and Senior Management',
            '3.3. Policy relating to the Remuneration for the KMP and Senior Management Personnel'
        ]

    elif file == 'Inactive_TradingAccount_Policy_version3.pdf':
        topic = '1. Reactivation of inactive Accounts'
        headers = [
            'A. Category-1-within 12 months after being flagged as Inactive (Activation without IPV) ',
            'B. Category-2- More than 12 months after being flagged as inactive (with IPV)'
        ]

    results = []

    for i in range(len(headers)):

        try:

            start_index = text.index(headers[i])

            end_index = text.index(headers[i + 1]) if i + 1 < len(headers) else len(text)

            content = text[start_index:end_index].strip()

            print(f"start_index: {start_index}, end_index: {end_index}")

            results.append({

                "metadata": {
                    
                    "source":file,
                    
                    "topic": topic,

                         "subtopic": headers[i],

                         "subtopic_2": ''},

                'content': content
            })
            
        except ValueError:

            print(f"[!] Header not found: {headers[i]} — skipped.")

            continue

    return results


