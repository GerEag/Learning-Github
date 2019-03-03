import java.util.Date;

public class Sensor {

    int sensorID;
    int applicationID;
    int csnetID;
    String sensorName;
    String lastCommDate;
    String nextCommDate;
    String lastMsgGUID;
    int powerSrcId;
    int status;
    boolean canUpdate;
    double humidityValue;
    double tempValue;
    double batteryLevel;
    double signalStrength;
    boolean alertsActive;
    String checkDigit;
    int AccountID;
    int monnitAppID;



    public Sensor(int sensorID, int applicationID, int csnetID, String sensorName, String lastCommDate,
			String nextCommDate, String lastMsgGUID, int powerSrcId, int status, boolean canUpdate,
			double tempValue, double humidityValue, double batteryLevel, double signalStrength,
			boolean alertsActive, String checkDigit, int accountID, int monnitAppID) {
    	this.sensorID = sensorID;
        this.applicationID = applicationID;
        this.csnetID = csnetID;
        this.sensorName = sensorName;
        this.lastCommDate = lastCommDate;
        this.nextCommDate = nextCommDate;
        this.lastMsgGUID = lastMsgGUID;
        this.powerSrcId = powerSrcId;
        this.status = status;
        this.canUpdate = canUpdate;
        this.tempValue = tempValue;
        this.humidityValue = humidityValue;
        this.batteryLevel = batteryLevel;
        this.signalStrength = signalStrength;
        this.alertsActive = alertsActive;
        this.checkDigit = checkDigit;
        this.AccountID = accountID;
        this.monnitAppID = monnitAppID;
		// TODO Auto-generated constructor stub
	}

	public double getTempValue() {
		return tempValue;
	}

	public void setTempValue(double tempValue) {
		this.tempValue = tempValue;
	}

	public int getSensorID() {
        return sensorID;
    }

    public void setSensorID(int sensorID) {
        this.sensorID = sensorID;
    }

    public int getApplicationID() {
        return applicationID;
    }

    public void setApplicationID(int applicationID) {
        this.applicationID = applicationID;
    }

    public int getCsnetID() {
        return csnetID;
    }

    public void setCsnetID(int csnetID) {
        this.csnetID = csnetID;
    }

    public String getSensorName() {
        return sensorName;
    }

    public void setSensorName(String sensorName) {
        this.sensorName = sensorName;
    }

    public String getLastCommDate() {
        return lastCommDate;
    }

    public void setLastCommDate(String lastCommDate) {
        this.lastCommDate = lastCommDate;
    }

    public String getNextCommDate() {
        return nextCommDate;
    }

    public void setNextCommDate(String nextCommDate) {
        this.nextCommDate = nextCommDate;
    }

    public String getLastMsgGUID() {
        return lastMsgGUID;
    }

    public void setLastMsgGUID(String lastMsgGUID) {
        this.lastMsgGUID = lastMsgGUID;
    }

    public int getPowerSrcId() {
        return powerSrcId;
    }

    public void setPowerSrcId(int powerSrcId) {
        this.powerSrcId = powerSrcId;
    }

    public int getStatus() {
        return status;
    }

    public void setStatus(int status) {
        this.status = status;
    }

    public boolean isCanUpdate() {
        return canUpdate;
    }

    public void setCanUpdate(boolean canUpdate) {
        this.canUpdate = canUpdate;
    }

    public double getHumidityValue() {
        return humidityValue;
    }

    public void setHumidityValue(double humidityValue) {
        this.humidityValue = humidityValue;
    }

    public double getBatteryLevel() {
        return batteryLevel;
    }

    public void setBatteryLevel(double batteryLevel) {
        this.batteryLevel = batteryLevel;
    }

    public double getSignalStrength() {
        return signalStrength;
    }

    public void setSignalStrength(double signalStrength) {
        this.signalStrength = signalStrength;
    }

    public boolean isAlertsActive() {
        return alertsActive;
    }

    public void setAlertsActive(boolean alertsActive) {
        this.alertsActive = alertsActive;
    }

    public String getCheckDigit() {
        return checkDigit;
    }

    public void setCheckDigit(String checkDigit) {
        this.checkDigit = checkDigit;
    }

    public int getAccountID() {
        return AccountID;
    }

    public void setAccountID(int accountID) {
        AccountID = accountID;
    }

    public int getMonnitAppID() {
        return monnitAppID;
    }

    public void setMonnitAppID(int monnitAppID) {
        this.monnitAppID = monnitAppID;
    }
}
