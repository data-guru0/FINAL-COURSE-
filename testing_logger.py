from src.logger import get_logger
from src.custom_exception import CustomException
import sys

logger = get_logger(__name__)

def divide_numbers(a,b):
    try:
        result = a/b
        logger.info("Dividing the numbers")
        return result
    except Exception as e:
        logger.error("Error Occured" )
        raise CustomException("Division by zero" , sys)
    
if __name__ == "__main__":
    try:
        logger.info("Starting main program")
        divide_numbers(10,0)
    except CustomException as ce:
        logger.error(str(ce))
    finally:
        logger.info("end pf program")
    

    